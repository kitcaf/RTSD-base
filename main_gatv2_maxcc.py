"""
当前实验:
    1. 骨干继续使用 2 层 GATv2
    2. 输入图保持全图
    3. 删除 rank loss，仅保留 all_infected 上的 BCE 监督
    4. 主干只消费基础 + 最大CC相对特征
    5. 5个核心角色特征仅在最大CC内部做 residual re-scoring
    6. 评测、阈值搜索和模型选择继续保持 all_infected 单视角

设计动机:
    - 已验证 max_cc 内的 rank loss 无效甚至有害
    - dataAnaly.md 表明源点更像“中心型启动者”，而非单纯热点 hub
    - 因此当前实验聚焦于“最大CC内部相对角色重打分”
"""
from __future__ import annotations

import os
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader

from config import (
    BATCH_SIZE,
    DATASETS,
    DATASET_NAMES,
    DATASET_IDX,
    DEFAULT_POS_WEIGHT,
    DEVICE,
    EPOCHS,
    GFRR_ARCH_CONFIGS,
    LR,
    MAX_CC_ROLE_RESCORER_DROPOUT,
    MAX_CC_ROLE_RESIDUAL_SCALE,
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
from main_gatv2_candidate_domain import ALL_INFECTED, build_loader, evaluate_candidate_mode, find_optimal_threshold
from metrics_utils import precompute_shortest_paths
from model.maxcc_role_gatv2 import MaxCCRelativeRoleGATv2
from utils import compute_ranking_focused_score, log_print, setup_seed, setup_training_logger


CHECKPOINT_DIR = "checkpoints_maxcc"
LOG_SUFFIX = "gatv2_maxcc_log.txt"
SAVE_TAG = "maxcc_role_rescore"
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
    primary_role_feature_names: Tuple[str, ...]
    primary_role_feature_indices: Tuple[int, ...]
    role_hidden_dim: int
    role_dropout: float
    role_residual_scale: float
    recall_k_values: Tuple[int, ...]
    model_selection_recall_k: int
    model_selection_weights: Dict[str, float]


@dataclass
class TrainingResult:
    best_epoch: int
    best_threshold: float
    best_val_f1: float
    best_val_score: float
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
        primary_role_feature_names=tuple(FeatureEngineerGFRR.PRIMARY_ROLE_FEATURE_NAMES),
        primary_role_feature_indices=tuple(
            FeatureEngineerGFRR.get_feature_indices(FeatureEngineerGFRR.PRIMARY_ROLE_FEATURE_NAMES)
        ),
        role_hidden_dim=max(16, arch_config.get("hidden_dim", 64) // 2),
        role_dropout=MAX_CC_ROLE_RESCORER_DROPOUT,
        role_residual_scale=MAX_CC_ROLE_RESIDUAL_SCALE,
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


def build_bce_criterion(pos_weight: float, device: torch.device) -> nn.BCEWithLogitsLoss:
    return nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], device=device))


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.BCEWithLogitsLoss,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> Tuple[float, Dict[str, float]]:
    model.train()

    total_loss = 0.0
    loss_components = {"bce": 0.0}
    num_batches = 0

    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()

        train_mask = batch.train_mask if hasattr(batch, "train_mask") else batch.loss_mask
        if train_mask.sum().item() == 0:
            continue

        logits = model(batch)
        batch_loss = criterion(logits[train_mask], batch.y[train_mask])
        batch_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += float(batch_loss.item())
        loss_components["bce"] += float(batch_loss.item())
        num_batches += 1

    avg_loss = total_loss / max(num_batches, 1)
    loss_components["bce"] /= max(num_batches, 1)
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

    model = MaxCCRelativeRoleGATv2(
        num_features=train_set[0].x.size(-1),
        backbone_feature_dim=config.backbone_feature_dim,
        role_feature_indices=config.primary_role_feature_indices,
        hidden_dim=config.hidden_dim,
        num_layers=config.num_layers,
        heads=config.heads,
        dropout=config.dropout,
        role_hidden_dim=config.role_hidden_dim,
        role_dropout=config.role_dropout,
        role_residual_scale=config.role_residual_scale,
    ).to(DEVICE)

    criterion = build_bce_criterion(config.pos_weight, DEVICE)
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

    for epoch in range(config.epochs):
        avg_loss, loss_comp = train_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=DEVICE,
        )

        best_threshold_candidate, _ = find_optimal_threshold(
            model=model,
            loader=val_loader,
            device=DEVICE,
            candidate_mode=EVAL_CANDIDATE,
        )
        val_metrics = evaluate_candidate_mode(
            model=model,
            loader=val_loader,
            device=DEVICE,
            threshold=best_threshold_candidate,
            recall_k_values=config.recall_k_values,
            dist_matrix=dist_matrix,
            candidate_mode=EVAL_CANDIDATE,
        )
        val_score = compute_ranking_focused_score(
            val_metrics,
            recall_k=config.model_selection_recall_k,
            weights=config.model_selection_weights,
        )

        if val_score > best_val_score:
            best_val_score = val_score
            best_val_f1 = val_metrics["f1"]
            best_threshold = best_threshold_candidate
            best_epoch = epoch + 1
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "threshold": best_threshold,
                    "epoch": best_epoch,
                    "val_f1": best_val_f1,
                    "val_score": best_val_score,
                },
                checkpoint_path,
            )

        recall_summary = " | ".join(
            f"R@{k}: {val_metrics[f'recall@{k}']:.3f}" for k in config.recall_k_values
        )
        log_print(
            logger,
            f"Epoch {epoch + 1:03d} | "
            f"Loss: {avg_loss:.4f} (BCE:{loss_comp['bce']:.3f}) | "
            f"Val Score: {val_score:.4f} | Val F1: {val_metrics['f1']:.4f} | "
            f"MAP: {val_metrics['map']:.4f} | P@K: {val_metrics['p@k_true']:.4f} | {recall_summary}",
        )

    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])

    return model, TrainingResult(
        best_epoch=best_epoch,
        best_threshold=best_threshold,
        best_val_f1=best_val_f1,
        best_val_score=best_val_score,
        checkpoint_path=checkpoint_path,
    )


def print_summary(logger, data_name: str, metrics: Dict[str, float], training_result: TrainingResult) -> None:
    log_print(logger, "\n" + "=" * 60)
    log_print(logger, f"[*] 最大CC相对角色重打分测试结果 - {data_name}")
    log_print(logger, "=" * 60)
    log_print(
        logger,
        f"[*] 模型选择基准: {EVAL_CANDIDATE} | "
        f"Best Epoch={training_result.best_epoch}, "
        f"Val Score={training_result.best_val_score:.4f}, "
        f"Val F1={training_result.best_val_f1:.4f}, "
        f"Threshold={training_result.best_threshold:.3f}",
    )
    log_print(logger, f"[*] AUC       : {metrics['auc']:.4f}")
    log_print(logger, f"[*] Precision : {metrics['precision']:.4f}")
    log_print(logger, f"[*] Recall    : {metrics['recall']:.4f}")
    log_print(logger, f"[*] F1-Score  : {metrics['f1']:.4f}")
    for k in RECALL_K_VALUES:
        log_print(logger, f"[*] Recall@{k:<2} : {metrics[f'recall@{k}']:.4f}")
    log_print(logger, f"[*] MAP       : {metrics['map']:.4f}")
    log_print(logger, f"[*] P@K_true  : {metrics['p@k_true']:.4f}")
    log_print(logger, f"[*] AED       : {metrics['aed']:.4f}")
    log_print(logger, "=" * 60)


def main() -> None:
    config = build_experiment_config()
    setup_seed(SEED)

    data_name = DATASET_NAMES[config.dataset_idx]
    logger = setup_training_logger(log_name=f"{data_name}_{LOG_SUFFIX}")

    log_print(logger, "=" * 60)
    log_print(logger, "[*] 实验: 最大CC内相对角色重打分")
    log_print(logger, f"[*] 数据集: {data_name}")
    log_print(logger, f"[*] 设备: {DEVICE}")
    log_print(logger, "[*] 输入图: 全图")
    log_print(logger, "[*] 监督作用域: all_infected(train_mask)")
    log_print(logger, "[*] Rank Loss: 已删除")
    log_print(logger, "[*] 角色特征生成: feature_engineering_gfrr.py")
    log_print(logger, f"[*] 特征缓存版本: {FeatureEngineerGFRR.FEATURE_CACHE_VERSION}")
    log_print(logger, "[*] 评测作用域: all_infected")
    log_print(logger, f"[*] 仅最终快照: {config.final_only}")
    log_print(
        logger,
        f"[*] Backbone: hidden_dim={config.hidden_dim}, num_layers={config.num_layers}, dropout={config.dropout}",
    )
    log_print(
        logger,
        f"[*] Loss: pos_weight={config.pos_weight}, lr={config.lr}, weight_decay={config.weight_decay}",
    )
    log_print(
        logger,
        f"[*] Role Rescorer: backbone_feature_dim={config.backbone_feature_dim}, "
        f"role_hidden_dim={config.role_hidden_dim}, role_dropout={config.role_dropout}, "
        f"residual_scale={config.role_residual_scale}",
    )
    log_print(
        logger,
        f"[*] 模型选择: RankingScore (MAP/P@K_true/Recall@{config.model_selection_recall_k}/F1/AED)",
    )
    log_print(logger, "=" * 60)

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

    current_dim = dataset[0].x.size(1) if dataset else engineer.get_num_features()
    log_print(logger, f"[*] 当前输入特征维度: {current_dim}")
    log_print(logger, f"[*] GAT主干使用特征维度: {config.backbone_feature_dim}")
    log_print(logger, "[*] 当前阶段核心角色特征:")
    for feature_name in config.primary_role_feature_names:
        log_print(logger, f"    - {feature_name}")
    if current_dim < max(config.primary_role_feature_indices) + 1:
        raise RuntimeError("角色特征索引超出当前输入维度，请检查特征缓存或特征工程配置。")

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
    test_metrics = evaluate_candidate_mode(
        model=model,
        loader=test_loader,
        device=DEVICE,
        threshold=training_result.best_threshold,
        recall_k_values=config.recall_k_values,
        dist_matrix=dist_matrix,
        candidate_mode=EVAL_CANDIDATE,
    )
    print_summary(
        logger=logger,
        data_name=data_name,
        metrics=test_metrics,
        training_result=training_result,
    )


if __name__ == "__main__":
    main()
