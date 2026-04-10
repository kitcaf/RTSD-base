"""
当前实验:
    1. 骨干继续使用 2 层 GATv2
    2. 输入图保持全图
    3. 删除 rank loss 和阶段1角色残差重打分
    4. 升级为共享 backbone + 并行 count head 的多任务结构
    5. 验证与测试采用 count-guided 推理，显式使用 K_hat 重排最大CC内候选节点

设计动机:
    - 阶段1说明单纯 max_cc 内逐点重打分不足以稳定提升 MAP / P@K_true
    - 当前真正缺失的是“该在最大CC中选几个源点”的建模
    - 因此阶段2先补齐 count estimation，再观察对最终 all_infected 评测的影响
"""
from __future__ import annotations

import os
import random
from dataclasses import dataclass, replace
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from torch_geometric.loader import DataLoader

from config import (
    BATCH_SIZE,
    COUNT_GUIDED_MIN_SELECTION,
    COUNT_GUIDED_NON_TOPK_DECAY,
    COUNT_GUIDED_OUTSIDE_MAXCC_DECAY,
    COUNT_GUIDED_TOPK_BOOST,
    COUNT_HEAD_DROPOUT,
    COUNT_LOSS_WEIGHT,
    COUNT_MODEL_SELECTION_WEIGHT,
    DATASETS,
    DATASET_NAMES,
    DATASET_IDX,
    DEFAULT_POS_WEIGHT,
    DEVICE,
    EPOCHS,
    GFRR_ARCH_CONFIGS,
    LR,
    MAX_CC_GRAPH_FINAL_ONLY,
    MODEL_SELECTION_RECALL_K,
    MODEL_SELECTION_WEIGHTS,
    RECALL_K_VALUES,
    SEED,
    TRAIN_RATIO,
    VAL_RATIO,
    WEIGHT_DECAY,
    get_gfrr_loss_config,
)
from data_loader import load_raw_data
from feature_engineering_gfrr import FeatureEngineerGFRR
from main_gatv2_candidate_domain import ALL_INFECTED, build_loader
from metrics_utils import calculate_aed, calculate_map, calculate_precision_at_k, precompute_shortest_paths
from model.maxcc_count_gatv2 import MaxCCCountMultiTaskGATv2
from utils import (
    build_count_guided_scores,
    compute_count_metrics,
    compute_multitask_selection_score,
    get_max_cc_source_count_targets,
    log_print,
    setup_seed,
    setup_training_logger,
    unwrap_model_outputs,
)


CHECKPOINT_DIR = "checkpoints_maxcc"
LOG_SUFFIX = "gatv2_maxcc_log.txt"
SAVE_TAG = "maxcc_count_multitask"
EVAL_CANDIDATE = ALL_INFECTED


@dataclass(frozen=True)
class ExperimentConfig:
    dataset_idx: int
    epochs: int
    lr: float
    weight_decay: float
    batch_size: int
    heads: int
    hidden_dim: int
    num_layers: int
    dropout: float
    final_only: bool
    pos_weight: float
    backbone_feature_dim: int
    count_hidden_dim: int
    count_dropout: float
    count_loss_weight: float
    count_model_selection_weight: float
    count_guided_topk_boost: float
    count_guided_non_topk_decay: float
    count_guided_outside_maxcc_decay: float
    count_guided_min_selection: int
    count_num_classes: int
    recall_k_values: Tuple[int, ...]
    model_selection_recall_k: int
    model_selection_weights: Dict[str, float]


@dataclass
class TrainingResult:
    best_epoch: int
    best_threshold: float
    best_val_f1: float
    best_val_score: float
    best_val_count_mae: float
    checkpoint_path: str


def build_experiment_config() -> ExperimentConfig:
    data_name = DATASET_NAMES[DATASET_IDX]
    arch_config = GFRR_ARCH_CONFIGS[data_name]
    loss_config = get_gfrr_loss_config()

    return ExperimentConfig(
        dataset_idx=DATASET_IDX,
        epochs=EPOCHS,
        lr=LR,
        weight_decay=WEIGHT_DECAY,
        batch_size=BATCH_SIZE,
        heads=4,
        hidden_dim=arch_config.get("hidden_dim", 64),
        num_layers=2,
        dropout=arch_config.get("dropout", 0.3),
        final_only=MAX_CC_GRAPH_FINAL_ONLY,
        pos_weight=loss_config.get("pos_weight", DEFAULT_POS_WEIGHT),
        backbone_feature_dim=FeatureEngineerGFRR.NON_ROLE_FEATURE_DIM,
        count_hidden_dim=max(16, arch_config.get("hidden_dim", 64) // 2),
        count_dropout=COUNT_HEAD_DROPOUT,
        count_loss_weight=COUNT_LOSS_WEIGHT,
        count_model_selection_weight=COUNT_MODEL_SELECTION_WEIGHT,
        count_guided_topk_boost=COUNT_GUIDED_TOPK_BOOST,
        count_guided_non_topk_decay=COUNT_GUIDED_NON_TOPK_DECAY,
        count_guided_outside_maxcc_decay=COUNT_GUIDED_OUTSIDE_MAXCC_DECAY,
        count_guided_min_selection=COUNT_GUIDED_MIN_SELECTION,
        count_num_classes=0,
        recall_k_values=tuple(RECALL_K_VALUES),
        model_selection_recall_k=MODEL_SELECTION_RECALL_K,
        model_selection_weights=dict(MODEL_SELECTION_WEIGHTS),
    )


def split_dataset_by_cascade(dataset, train_ratio: float, val_ratio: float) -> Tuple[List, List, List]:
    cascade_ids = sorted(set(int(data.cascade_id) for data in dataset))
    random.shuffle(cascade_ids)

    num_cascades = len(cascade_ids)
    train_end = int(num_cascades * train_ratio)
    val_end = int(num_cascades * (train_ratio + val_ratio))

    train_ids = set(cascade_ids[:train_end])
    val_ids = set(cascade_ids[train_end:val_end])
    test_ids = set(cascade_ids[val_end:])

    train_set = [data for data in dataset if int(data.cascade_id) in train_ids]
    val_set = [data for data in dataset if int(data.cascade_id) in val_ids and bool(data.is_final)]
    test_set = [data for data in dataset if int(data.cascade_id) in test_ids and bool(data.is_final)]
    return train_set, val_set, test_set


def infer_count_num_classes(dataset: List) -> int:
    max_count = 0
    for sample in dataset:
        if not hasattr(sample, "max_cc_mask"):
            raise ValueError("dataset sample is missing max_cc_mask")
        max_count = max(
            max_count,
            int((sample.y.bool() & sample.max_cc_mask.bool()).sum().item()),
        )
    return max(max_count + 1, 2)


def build_bce_criterion(pos_weight: float, device: torch.device) -> nn.BCEWithLogitsLoss:
    return nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], device=device))


def collect_count_guided_outputs(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    config: ExperimentConfig,
) -> List[Dict[str, np.ndarray | int]]:
    model.eval()
    collected_outputs: List[Dict[str, np.ndarray | int]] = []

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            train_mask = batch.train_mask.bool()
            if train_mask.sum().item() == 0:
                continue

            node_logits, count_logits = unwrap_model_outputs(model(batch))
            node_probs = torch.sigmoid(node_logits).detach().cpu().numpy()

            candidate_mask_np = train_mask.detach().cpu().numpy()
            candidate_node_indices = np.where(candidate_mask_np)[0]
            candidate_scores = node_probs[candidate_mask_np]
            max_cc_mask_full = batch.max_cc_mask.bool().detach().cpu().numpy()

            raw_predicted_count = int(torch.argmax(count_logits, dim=-1)[0].item())
            max_cc_candidate_count = int((train_mask & batch.max_cc_mask.bool()).sum().item())
            clipped_predicted_count = min(raw_predicted_count, max_cc_candidate_count)

            guided_scores = build_count_guided_scores(
                candidate_scores=candidate_scores,
                candidate_node_indices=candidate_node_indices,
                max_cc_mask_full=max_cc_mask_full,
                predicted_count=clipped_predicted_count,
                topk_boost=config.count_guided_topk_boost,
                non_topk_decay=config.count_guided_non_topk_decay,
                outside_maxcc_decay=config.count_guided_outside_maxcc_decay,
                min_selection=config.count_guided_min_selection,
            )

            true_count = int(get_max_cc_source_count_targets(batch, config.count_num_classes)[0].item())
            y_true = batch.y[train_mask].detach().cpu().numpy()

            collected_outputs.append(
                {
                    "y_true": y_true,
                    "scores": guided_scores,
                    "node_indices": candidate_node_indices,
                    "predicted_count": clipped_predicted_count,
                    "true_count": true_count,
                }
            )

    return collected_outputs


def find_optimal_threshold_from_outputs(collected_outputs: List[Dict[str, np.ndarray | int]]) -> Tuple[float, float]:
    thresholds = np.arange(0.1, 0.95, 0.01)
    if not collected_outputs:
        return 0.5, 0.0

    best_threshold = 0.5
    best_f1 = 0.0

    for threshold in thresholds:
        f1_values = []
        for item in collected_outputs:
            y_true = item["y_true"]
            y_scores = item["scores"]
            y_pred = (y_scores > threshold).astype(int)
            if y_pred.sum() == 0 and len(y_scores) > 0:
                y_pred[np.argmax(y_scores)] = 1
            f1_values.append(f1_score(y_true, y_pred, zero_division=0))

        avg_f1 = float(np.mean(f1_values)) if f1_values else 0.0
        if avg_f1 > best_f1:
            best_f1 = avg_f1
            best_threshold = float(threshold)

    return best_threshold, best_f1


def evaluate_count_guided_outputs(
    collected_outputs: List[Dict[str, np.ndarray | int]],
    threshold: float,
    recall_k_values: Tuple[int, ...],
    dist_matrix,
) -> Dict[str, float]:
    precision_list, recall_list, f1_list, auc_list = [], [], [], []
    recall_at_k_lists = {int(k): [] for k in recall_k_values}
    map_list, pk_list, aed_list = [], [], []
    predicted_counts, true_counts = [], []

    for item in collected_outputs:
        y_true = item["y_true"]
        y_scores = item["scores"]
        node_indices = item["node_indices"]
        predicted_counts.append(int(item["predicted_count"]))
        true_counts.append(int(item["true_count"]))

        try:
            if len(np.unique(y_true)) > 1:
                auc_list.append(roc_auc_score(y_true, y_scores))
        except ValueError:
            pass

        num_sources = int(y_true.sum())
        num_candidates = len(y_scores)
        if num_sources > 0:
            sorted_idx = np.argsort(-y_scores)
            for k in recall_at_k_lists:
                top_k = sorted_idx[:k]
                hits = y_true[top_k].sum()
                recall_at_k_lists[k].append(hits / num_sources)

            map_list.append(calculate_map(y_scores, y_true, num_candidates))
            pk_list.append(calculate_precision_at_k(y_scores, y_true, num_sources))
            aed_list.append(
                calculate_aed(
                    y_scores,
                    y_true,
                    dist_matrix=dist_matrix,
                    top_k=num_sources,
                    node_indices=node_indices,
                )
            )

        y_pred = (y_scores > threshold).astype(int)
        if y_pred.sum() == 0 and len(y_scores) > 0:
            y_pred[np.argmax(y_scores)] = 1

        precision_list.append(precision_score(y_true, y_pred, zero_division=0))
        recall_list.append(recall_score(y_true, y_pred, zero_division=0))
        f1_list.append(f1_score(y_true, y_pred, zero_division=0))

    metrics = {
        "auc": float(np.mean(auc_list)) if auc_list else 0.0,
        "precision": float(np.mean(precision_list)) if precision_list else 0.0,
        "recall": float(np.mean(recall_list)) if recall_list else 0.0,
        "f1": float(np.mean(f1_list)) if f1_list else 0.0,
        "map": float(np.mean(map_list)) if map_list else 0.0,
        "p@k_true": float(np.mean(pk_list)) if pk_list else 0.0,
        "aed": float(np.mean(aed_list)) if aed_list else 0.0,
    }
    metrics.update(compute_count_metrics(predicted_counts, true_counts))
    for k in recall_at_k_lists:
        metrics[f"recall@{k}"] = float(np.mean(recall_at_k_lists[k])) if recall_at_k_lists[k] else 0.0

    return metrics


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion_node: nn.BCEWithLogitsLoss,
    criterion_count: nn.CrossEntropyLoss,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    config: ExperimentConfig,
) -> Tuple[float, Dict[str, float]]:
    model.train()

    total_loss = 0.0
    loss_components = {"node_bce": 0.0, "count_ce": 0.0}
    num_batches = 0

    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()

        train_mask = batch.train_mask if hasattr(batch, "train_mask") else batch.loss_mask
        if train_mask.sum().item() == 0:
            continue

        node_logits, count_logits = unwrap_model_outputs(model(batch))
        node_loss = criterion_node(node_logits[train_mask], batch.y[train_mask])
        count_targets = get_max_cc_source_count_targets(batch, config.count_num_classes)
        count_loss = criterion_count(count_logits, count_targets)
        total_batch_loss = node_loss + config.count_loss_weight * count_loss

        total_batch_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += float(total_batch_loss.item())
        loss_components["node_bce"] += float(node_loss.item())
        loss_components["count_ce"] += float(count_loss.item())
        num_batches += 1

    avg_loss = total_loss / max(num_batches, 1)
    for key in loss_components:
        loss_components[key] /= max(num_batches, 1)
    return avg_loss, loss_components


def train_model_once(
    train_set: List,
    val_set: List,
    config: ExperimentConfig,
    data_name: str,
    dist_matrix,
    logger,
) -> Tuple[nn.Module, TrainingResult]:
    setup_seed(SEED)

    train_loader = build_loader(train_set, batch_size=config.batch_size, shuffle=True)
    val_loader = build_loader(val_set, batch_size=1, shuffle=False)

    model = MaxCCCountMultiTaskGATv2(
        num_features=train_set[0].x.size(-1),
        backbone_feature_dim=config.backbone_feature_dim,
        count_num_classes=config.count_num_classes,
        hidden_dim=config.hidden_dim,
        num_layers=config.num_layers,
        heads=config.heads,
        dropout=config.dropout,
        count_hidden_dim=config.count_hidden_dim,
        count_dropout=config.count_dropout,
    ).to(DEVICE)

    criterion_node = build_bce_criterion(config.pos_weight, DEVICE)
    criterion_count = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    checkpoint_path = os.path.join(
        CHECKPOINT_DIR,
        f"gatv2_{data_name}_{SAVE_TAG}_best.pt",
    )

    best_epoch = 0
    best_threshold = 0.5
    best_val_f1 = 0.0
    best_val_score = float("-inf")
    best_val_count_mae = float("inf")

    for epoch in range(config.epochs):
        avg_loss, loss_comp = train_epoch(
            model=model,
            loader=train_loader,
            criterion_node=criterion_node,
            criterion_count=criterion_count,
            optimizer=optimizer,
            device=DEVICE,
            config=config,
        )

        val_outputs = collect_count_guided_outputs(
            model=model,
            loader=val_loader,
            device=DEVICE,
            config=config,
        )
        best_threshold_candidate, candidate_val_f1 = find_optimal_threshold_from_outputs(val_outputs)
        val_metrics = evaluate_count_guided_outputs(
            collected_outputs=val_outputs,
            threshold=best_threshold_candidate,
            recall_k_values=config.recall_k_values,
            dist_matrix=dist_matrix,
        )
        val_score = compute_multitask_selection_score(
            val_metrics,
            recall_k=config.model_selection_recall_k,
            ranking_weights=config.model_selection_weights,
            count_weight=config.count_model_selection_weight,
        )

        if val_score > best_val_score:
            best_val_score = val_score
            best_val_f1 = candidate_val_f1
            best_val_count_mae = val_metrics["count_mae"]
            best_threshold = best_threshold_candidate
            best_epoch = epoch + 1
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "threshold": best_threshold,
                    "epoch": best_epoch,
                    "val_f1": best_val_f1,
                    "val_score": best_val_score,
                    "val_count_mae": best_val_count_mae,
                },
                checkpoint_path,
            )

        recall_summary = " | ".join(
            f"R@{k}: {val_metrics[f'recall@{k}']:.3f}" for k in config.recall_k_values
        )
        log_print(
            logger,
            f"Epoch {epoch + 1:03d} | "
            f"Loss: {avg_loss:.4f} (Node:{loss_comp['node_bce']:.3f}, Count:{loss_comp['count_ce']:.3f}) | "
            f"Val Score: {val_score:.4f} | Val F1: {val_metrics['f1']:.4f} | "
            f"Count MAE: {val_metrics['count_mae']:.4f} | "
            f"MAP: {val_metrics['map']:.4f} | P@K: {val_metrics['p@k_true']:.4f} | {recall_summary}",
        )

    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])

    return model, TrainingResult(
        best_epoch=best_epoch,
        best_threshold=best_threshold,
        best_val_f1=best_val_f1,
        best_val_score=best_val_score,
        best_val_count_mae=best_val_count_mae,
        checkpoint_path=checkpoint_path,
    )


def print_summary(logger, data_name: str, metrics: Dict[str, float], training_result: TrainingResult) -> None:
    log_print(logger, "\n" + "=" * 60)
    log_print(logger, f"[*] 最大CC源点个数估计测试结果 - {data_name}")
    log_print(logger, "=" * 60)
    log_print(
        logger,
        f"[*] 模型选择基准: {EVAL_CANDIDATE} | "
        f"Best Epoch={training_result.best_epoch}, "
        f"Val Score={training_result.best_val_score:.4f}, "
        f"Val F1={training_result.best_val_f1:.4f}, "
        f"Val Count MAE={training_result.best_val_count_mae:.4f}, "
        f"Threshold={training_result.best_threshold:.3f}",
    )
    log_print(logger, f"[*] AUC         : {metrics['auc']:.4f}")
    log_print(logger, f"[*] Precision   : {metrics['precision']:.4f}")
    log_print(logger, f"[*] Recall      : {metrics['recall']:.4f}")
    log_print(logger, f"[*] F1-Score    : {metrics['f1']:.4f}")
    for k in RECALL_K_VALUES:
        log_print(logger, f"[*] Recall@{k:<2}   : {metrics[f'recall@{k}']:.4f}")
    log_print(logger, f"[*] MAP         : {metrics['map']:.4f}")
    log_print(logger, f"[*] P@K_true    : {metrics['p@k_true']:.4f}")
    log_print(logger, f"[*] AED         : {metrics['aed']:.4f}")
    log_print(logger, f"[*] Count MAE   : {metrics['count_mae']:.4f}")
    log_print(logger, f"[*] Count Acc   : {metrics['count_acc']:.4f}")
    log_print(logger, f"[*] Count ±1 Acc: {metrics['count_within_1']:.4f}")
    log_print(logger, "=" * 60)


def main() -> None:
    config = build_experiment_config()
    setup_seed(SEED)

    data_name = DATASET_NAMES[config.dataset_idx]
    logger = setup_training_logger(log_name=f"{data_name}_{LOG_SUFFIX}")

    adjacency_matrix, influence_matrices = load_raw_data(DATASETS[config.dataset_idx])
    engineer = FeatureEngineerGFRR(adjacency_matrix)
    dataset = engineer.generate_dataset(
        influence_matrices,
        cache_name=data_name,
        use_gfrr=False,
        use_max_cc_graph=False,
        final_only=config.final_only,
        require_source_in_graph=False,
    )
    config = replace(config, count_num_classes=infer_count_num_classes(dataset))

    log_print(logger, "=" * 60)
    log_print(logger, "[*] 实验: 最大CC源点个数估计 + count-guided 推理")
    log_print(logger, f"[*] 数据集: {data_name}")
    log_print(logger, f"[*] 设备: {DEVICE}")
    log_print(logger, "[*] 输入图: 全图")
    log_print(logger, "[*] 监督作用域: all_infected(train_mask) + max_cc count target")
    log_print(logger, "[*] Rank Loss: 已删除")
    log_print(logger, "[*] 角色残差重打分: 已移除")
    log_print(logger, f"[*] 特征缓存版本: {FeatureEngineerGFRR.FEATURE_CACHE_VERSION}")
    log_print(logger, "[*] 评测作用域: all_infected")
    log_print(logger, f"[*] 仅最终快照: {config.final_only}")
    log_print(
        logger,
        f"[*] Backbone: hidden_dim={config.hidden_dim}, num_layers={config.num_layers}, dropout={config.dropout}",
    )
    log_print(
        logger,
        f"[*] Loss: pos_weight={config.pos_weight}, lr={config.lr}, weight_decay={config.weight_decay}, "
        f"count_loss_weight={config.count_loss_weight}",
    )
    log_print(
        logger,
        f"[*] Count Head: classes={config.count_num_classes}, hidden_dim={config.count_hidden_dim}, "
        f"dropout={config.count_dropout}",
    )
    log_print(
        logger,
        f"[*] Count-Guided: topk_boost={config.count_guided_topk_boost}, "
        f"non_topk_decay={config.count_guided_non_topk_decay}, "
        f"outside_maxcc_decay={config.count_guided_outside_maxcc_decay}, "
        f"min_selection={config.count_guided_min_selection}",
    )
    log_print(
        logger,
        f"[*] 模型选择: RankingScore + CountMAE(recall@{config.model_selection_recall_k})",
    )
    log_print(logger, "=" * 60)

    current_dim = dataset[0].x.size(1) if dataset else engineer.get_num_features()
    log_print(logger, f"[*] 当前输入特征维度: {current_dim}")
    log_print(logger, f"[*] GAT主干使用特征维度: {config.backbone_feature_dim}")
    if current_dim < config.backbone_feature_dim:
        raise RuntimeError("当前输入特征维度小于主干要求维度，请检查特征工程配置。")

    train_set, val_set, test_set = split_dataset_by_cascade(
        dataset,
        train_ratio=TRAIN_RATIO,
        val_ratio=VAL_RATIO,
    )
    if not train_set or not val_set or not test_set:
        raise RuntimeError("Train/Val/Test 至少有一个为空，请检查数据划分。")

    train_snapshot_desc = "仅最终快照" if config.final_only else "含所有快照"
    log_print(
        logger,
        f"[*] 数据划分: Train={len(train_set)} ({train_snapshot_desc}), "
        f"Val={len(val_set)} (仅最终快照), Test={len(test_set)} (仅最终快照)",
    )

    log_print(logger, "[*] 预计算最短路径矩阵...")
    dist_matrix = precompute_shortest_paths(adjacency_matrix)

    model, training_result = train_model_once(
        train_set=train_set,
        val_set=val_set,
        config=config,
        data_name=data_name,
        dist_matrix=dist_matrix,
        logger=logger,
    )

    test_loader = build_loader(test_set, batch_size=1, shuffle=False)
    test_outputs = collect_count_guided_outputs(
        model=model,
        loader=test_loader,
        device=DEVICE,
        config=config,
    )
    test_metrics = evaluate_count_guided_outputs(
        collected_outputs=test_outputs,
        threshold=training_result.best_threshold,
        recall_k_values=config.recall_k_values,
        dist_matrix=dist_matrix,
    )
    print_summary(
        logger=logger,
        data_name=data_name,
        metrics=test_metrics,
        training_result=training_result,
    )


if __name__ == "__main__":
    main()
