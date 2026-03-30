"""
单模型候选域诊断实验:
    1. 训练一次: 全图输入, loss 作用在所有感染节点上
    2. 评测两次: 使用不同候选域计算指标
        - all_infected: 所有感染节点
        - max_cc: 最大感染连通块中的感染节点

额外诊断:
    - top-k_true 预测里, 最大CC外节点占比
    - 感染节点预测概率质量里, 最大CC外节点占比
"""
from __future__ import annotations

import os
import random
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GATv2Conv

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
    MAX_CC_GRAPH_FINAL_ONLY,
    RECALL_K_VALUES,
    SEED,
    TRAIN_RATIO,
    VAL_RATIO,
    WEIGHT_DECAY,
    get_gfrr_loss_config,
)
from data_loader import load_raw_data
from feature_engineering_gfrr import FeatureEngineerGFRR
from loss_gfrr import GFRRLoss
from metrics_utils import calculate_aed, calculate_map, calculate_precision_at_k, precompute_shortest_paths
from utils import log_print, setup_seed, setup_training_logger


ALL_INFECTED = "all_infected"
MAX_CC = "max_cc"
CANDIDATE_MODES = (ALL_INFECTED, MAX_CC)


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
    lambda_rank: float
    margin: float
    recall_k_values: Tuple[int, ...]


@dataclass
class TrainingResult:
    best_epoch: int
    best_threshold: float
    best_val_f1: float
    checkpoint_path: str


@dataclass
class CandidateEvalResult:
    candidate_mode: str
    metrics: Dict[str, float]
    threshold: float
    val_f1: float


@dataclass
class MaxCCDiagnostics:
    topk_true_outside_ratio_mean: float
    topk_true_outside_ratio_global: float
    outside_prob_mass_ratio_mean: float
    outside_prob_mass_ratio_global: float


class WholeGraphGATv2(nn.Module):
    """简洁的全图 GATv2 节点分类器。"""

    def __init__(
        self,
        num_features: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
        heads: int = 4,
        dropout: float = 0.3,
    ):
        super().__init__()

        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")

        self.dropout = dropout
        self.convs = nn.ModuleList()
        in_dim = num_features

        for _ in range(num_layers):
            self.convs.append(
                GATv2Conv(
                    in_channels=in_dim,
                    out_channels=hidden_dim,
                    heads=heads,
                    concat=False,
                    dropout=dropout,
                    add_self_loops=True,
                )
            )
            in_dim = hidden_dim

        self.classifier = nn.Linear(hidden_dim, 1)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        for layer_idx, conv in enumerate(self.convs):
            x_in = x
            x = conv(x, edge_index)
            x = F.elu(x)
            if layer_idx > 0 and x.shape == x_in.shape:
                x = x + x_in
            x = F.dropout(x, p=self.dropout, training=self.training)

        return self.classifier(x).squeeze(-1)


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
        lambda_rank=loss_config.get("lambda_rank", 0.1),
        margin=loss_config.get("margin", 0.15),
        recall_k_values=tuple(RECALL_K_VALUES),
    )


def get_candidate_mask(data, candidate_mode: str) -> torch.Tensor:
    infected_mask = data.train_mask.bool()
    if candidate_mode == ALL_INFECTED:
        return infected_mask
    if candidate_mode == MAX_CC:
        return infected_mask & data.max_cc_mask.bool()
    raise ValueError(f"Unknown candidate_mode: {candidate_mode}")


def summarize_max_cc_coverage(dataset) -> Dict[str, float]:
    if not dataset:
        return {
            "num_samples": 0,
            "covered_samples": 0,
            "coverage_ratio": 0.0,
            "avg_max_cc_ratio": 0.0,
        }

    covered = 0
    max_cc_ratios = []
    for data in dataset:
        source_mask = data.y.bool()
        if source_mask.sum().item() > 0 and data.max_cc_mask.bool()[source_mask].all().item():
            covered += 1
        max_cc_ratios.append(
            float(data.max_cc_ratio.item()) if torch.is_tensor(data.max_cc_ratio) else float(data.max_cc_ratio)
        )

    return {
        "num_samples": len(dataset),
        "covered_samples": covered,
        "coverage_ratio": covered / len(dataset),
        "avg_max_cc_ratio": float(np.mean(max_cc_ratios)) if max_cc_ratios else 0.0,
    }


def summarize_candidate_distribution(dataset, candidate_mode: str) -> Dict[str, float]:
    total_pos = 0
    total_neg = 0
    total_candidates = 0

    for data in dataset:
        mask = get_candidate_mask(data, candidate_mode)
        labels = data.y[mask]
        total_pos += int((labels == 1).sum().item())
        total_neg += int((labels == 0).sum().item())
        total_candidates += int(mask.sum().item())

    if total_candidates == 0:
        return {
            "total_candidates": 0,
            "total_pos": 0,
            "total_neg": 0,
            "pos_ratio": 0.0,
        }

    return {
        "total_candidates": total_candidates,
        "total_pos": total_pos,
        "total_neg": total_neg,
        "pos_ratio": total_pos / total_candidates,
    }


def split_dataset_by_cascade(dataset, train_ratio: float, val_ratio: float) -> Tuple[List, List, List]:
    cascade_ids = sorted(set(int(data.cascade_id) for data in dataset))
    random.shuffle(cascade_ids)

    n_cascades = len(cascade_ids)
    train_end = int(n_cascades * train_ratio)
    val_end = int(n_cascades * (train_ratio + val_ratio))

    train_ids = set(cascade_ids[:train_end])
    val_ids = set(cascade_ids[train_end:val_end])
    test_ids = set(cascade_ids[val_end:])

    train_set = [data for data in dataset if int(data.cascade_id) in train_ids]
    val_set = [data for data in dataset if int(data.cascade_id) in val_ids]
    test_set = [data for data in dataset if int(data.cascade_id) in test_ids]
    return train_set, val_set, test_set


def build_loader(dataset, batch_size: int, shuffle: bool = False) -> DataLoader:
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: GFRRLoss,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> Tuple[float, Dict[str, float]]:
    model.train()

    total_loss = 0.0
    loss_components = {"cls": 0.0, "bce": 0.0, "rank": 0.0}
    num_batches = 0

    for data in loader:
        data = data.to(device)

        optimizer.zero_grad()
        logits = model(data)
        loss_dict = criterion(
            logits,
            data.y,
            data.train_mask,
            k_inf=data.k_inf if hasattr(data, "k_inf") else None,
        )

        loss = loss_dict["total"]
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += float(loss.item())
        for key in loss_components:
            loss_components[key] += float(loss_dict[key].item())
        num_batches += 1

    avg_loss = total_loss / max(num_batches, 1)
    for key in loss_components:
        loss_components[key] /= max(num_batches, 1)

    return avg_loss, loss_components


@torch.no_grad()
def collect_candidate_outputs(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    candidate_mode: str,
) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    model.eval()
    outputs = []

    for data in loader:
        data = data.to(device)
        candidate_mask = get_candidate_mask(data, candidate_mode)
        if candidate_mask.sum().item() == 0:
            continue

        logits = model(data)
        probs = torch.sigmoid(logits[candidate_mask]).detach().cpu().numpy()
        y_true = data.y[candidate_mask].detach().cpu().numpy()
        node_indices = torch.where(candidate_mask)[0].detach().cpu().numpy()
        outputs.append((y_true, probs, node_indices))

    return outputs


def find_optimal_threshold(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    candidate_mode: str = ALL_INFECTED,
) -> Tuple[float, float]:
    thresholds = np.arange(0.1, 0.95, 0.01)
    candidate_outputs = collect_candidate_outputs(model, loader, device, candidate_mode)

    if not candidate_outputs:
        return 0.5, 0.0

    best_threshold = 0.5
    best_f1 = 0.0

    for threshold in thresholds:
        f1_list = []
        for y_true, y_scores, _ in candidate_outputs:
            y_pred = (y_scores > threshold).astype(int)
            if y_pred.sum() == 0 and len(y_scores) > 0:
                y_pred[np.argmax(y_scores)] = 1
            f1_list.append(f1_score(y_true, y_pred, zero_division=0))

        avg_f1 = float(np.mean(f1_list)) if f1_list else 0.0
        if avg_f1 > best_f1:
            best_f1 = avg_f1
            best_threshold = float(threshold)

    return best_threshold, best_f1


@torch.no_grad()
def evaluate_candidate_mode(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    threshold: float,
    recall_k_values: Iterable[int],
    dist_matrix: np.ndarray,
    candidate_mode: str,
) -> Dict[str, float]:
    model.eval()

    precision_list, recall_list, f1_list, auc_list = [], [], [], []
    recall_at_k_lists = {int(k): [] for k in recall_k_values}
    map_list, pk_list, aed_list = [], [], []

    for data in loader:
        data = data.to(device)
        candidate_mask = get_candidate_mask(data, candidate_mode)
        if candidate_mask.sum().item() == 0:
            continue

        logits = model(data)
        y_scores = torch.sigmoid(logits[candidate_mask]).detach().cpu().numpy()
        y_true = data.y[candidate_mask].detach().cpu().numpy()
        node_indices = torch.where(candidate_mask)[0].detach().cpu().numpy()

        try:
            if len(np.unique(y_true)) > 1:
                auc_list.append(roc_auc_score(y_true, y_scores))
        except Exception:
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

    for k in recall_at_k_lists:
        metrics[f"recall@{k}"] = float(np.mean(recall_at_k_lists[k])) if recall_at_k_lists[k] else 0.0

    return metrics


@torch.no_grad()
def compute_max_cc_diagnostics(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> MaxCCDiagnostics:
    model.eval()

    topk_outside_ratio_list = []
    prob_mass_outside_ratio_list = []

    total_topk_outside = 0
    total_topk = 0
    total_outside_mass = 0.0
    total_mass = 0.0

    for data in loader:
        data = data.to(device)
        infected_mask = data.train_mask.bool()
        if infected_mask.sum().item() == 0:
            continue

        logits = model(data)
        probs = torch.sigmoid(logits)

        infected_indices = torch.where(infected_mask)[0]
        infected_probs = probs[infected_mask]
        infected_labels = data.y[infected_mask]
        infected_max_cc = data.max_cc_mask.bool()[infected_indices]

        total_inf_prob = float(infected_probs.sum().item())
        outside_inf_prob = float(infected_probs[~infected_max_cc].sum().item())
        if total_inf_prob > 0:
            prob_mass_outside_ratio_list.append(outside_inf_prob / total_inf_prob)
        else:
            prob_mass_outside_ratio_list.append(0.0)

        total_outside_mass += outside_inf_prob
        total_mass += total_inf_prob

        k_true = int(infected_labels.sum().item())
        if k_true <= 0:
            continue

        k_true = min(k_true, infected_probs.numel())
        top_indices = torch.topk(infected_probs, k=k_true).indices
        top_is_outside = (~infected_max_cc[top_indices]).float()

        topk_outside_ratio_list.append(float(top_is_outside.mean().item()))
        total_topk_outside += int(top_is_outside.sum().item())
        total_topk += k_true

    return MaxCCDiagnostics(
        topk_true_outside_ratio_mean=float(np.mean(topk_outside_ratio_list)) if topk_outside_ratio_list else 0.0,
        topk_true_outside_ratio_global=(total_topk_outside / total_topk) if total_topk > 0 else 0.0,
        outside_prob_mass_ratio_mean=(
            float(np.mean(prob_mass_outside_ratio_list)) if prob_mass_outside_ratio_list else 0.0
        ),
        outside_prob_mass_ratio_global=(total_outside_mass / total_mass) if total_mass > 0 else 0.0,
    )


def train_model_once(
    train_set: List,
    val_set: List,
    config: ExperimentConfig,
    data_name: str,
    dist_matrix: np.ndarray,
    logger,
) -> Tuple[nn.Module, TrainingResult]:
    setup_seed(SEED)

    train_loader = build_loader(train_set, batch_size=config.batch_size, shuffle=True)
    val_loader = build_loader(val_set, batch_size=1, shuffle=False)

    model = WholeGraphGATv2(
        num_features=train_set[0].x.size(-1),
        hidden_dim=config.hidden_dim,
        num_layers=config.num_layers,
        heads=config.heads,
        dropout=config.dropout,
    ).to(DEVICE)

    criterion = GFRRLoss(
        pos_weight=config.pos_weight,
        lambda_rank=config.lambda_rank,
        margin=config.margin,
    ).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)

    ckpt_dir = "checkpoints_candidate_domain"
    os.makedirs(ckpt_dir, exist_ok=True)
    checkpoint_path = os.path.join(ckpt_dir, f"gatv2_{data_name}_shared_best.pt")

    train_stats = summarize_candidate_distribution(train_set, ALL_INFECTED)
    val_stats = summarize_candidate_distribution(val_set, ALL_INFECTED)
    log_print(logger, f"[*] 训练候选节点统计(all_infected): {train_stats}")
    log_print(logger, f"[*] 验证候选节点统计(all_infected): {val_stats}")

    best_epoch = 0
    best_threshold = 0.5
    best_val_f1 = 0.0

    for epoch in range(config.epochs):
        avg_loss, loss_comp = train_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=DEVICE,
        )

        threshold, _ = find_optimal_threshold(
            model=model,
            loader=val_loader,
            device=DEVICE,
            candidate_mode=ALL_INFECTED,
        )
        val_metrics = evaluate_candidate_mode(
            model=model,
            loader=val_loader,
            device=DEVICE,
            threshold=threshold,
            recall_k_values=config.recall_k_values,
            dist_matrix=dist_matrix,
            candidate_mode=ALL_INFECTED,
        )

        if val_metrics["f1"] > best_val_f1:
            best_val_f1 = val_metrics["f1"]
            best_threshold = threshold
            best_epoch = epoch + 1
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "threshold": best_threshold,
                    "epoch": best_epoch,
                    "val_f1": best_val_f1,
                },
                checkpoint_path,
            )

        recall_str = " | ".join(
            f"R@{k}: {val_metrics[f'recall@{k}']:.3f}" for k in config.recall_k_values
        )
        log_print(
            logger,
            f"Epoch {epoch + 1:03d} | "
            f"Loss: {avg_loss:.4f} "
            f"(BCE:{loss_comp['bce']:.3f}, Rank:{loss_comp['rank']:.3f}) | "
            f"Val F1(all_infected): {val_metrics['f1']:.4f} | {recall_str}",
        )

    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])

    training_result = TrainingResult(
        best_epoch=best_epoch,
        best_threshold=best_threshold,
        best_val_f1=best_val_f1,
        checkpoint_path=checkpoint_path,
    )
    return model, training_result


def print_comparison_summary(
    logger,
    data_name: str,
    eval_results: Dict[str, CandidateEvalResult],
    diagnostics: MaxCCDiagnostics,
    training_result: TrainingResult,
    recall_k_values: Iterable[int],
) -> None:
    log_print(logger, "\n" + "=" * 76)
    log_print(logger, f"[*] 候选域评测对比 - {data_name}")
    log_print(logger, "=" * 76)
    log_print(
        logger,
        f"[*] 单次训练最佳验证结果: Epoch={training_result.best_epoch}, "
        f"Val F1(all_infected)={training_result.best_val_f1:.4f}, "
        f"Threshold(all_infected@checkpoint)={training_result.best_threshold:.3f}",
    )
    log_print(logger, "[*] 候选域专属验证阈值:")
    for candidate_mode in CANDIDATE_MODES:
        result = eval_results[candidate_mode]
        log_print(
            logger,
            f"    {candidate_mode}: Threshold={result.threshold:.3f}, Val F1={result.val_f1:.4f}",
        )

    metric_names = ["auc", "precision", "recall", "f1", "map", "p@k_true", "aed"]
    metric_names.extend([f"recall@{k}" for k in recall_k_values])

    header = f"{'Metric':<16}{ALL_INFECTED:<16}{MAX_CC:<16}{'Delta(max_cc-all)':<18}"
    log_print(logger, header)
    log_print(logger, "-" * len(header))

    all_metrics = eval_results[ALL_INFECTED].metrics
    max_cc_metrics = eval_results[MAX_CC].metrics
    for metric_name in metric_names:
        all_value = all_metrics.get(metric_name, 0.0)
        max_cc_value = max_cc_metrics.get(metric_name, 0.0)
        delta = max_cc_value - all_value
        log_print(
            logger,
            f"{metric_name:<16}{all_value:<16.4f}{max_cc_value:<16.4f}{delta:<18.4f}",
        )

    log_print(logger, "\n[*] 最大CC诊断指标")
    log_print(
        logger,
        f"    top-k_true预测里最大CC外节点占比(mean): {diagnostics.topk_true_outside_ratio_mean:.4f}",
    )
    log_print(
        logger,
        f"    top-k_true预测里最大CC外节点占比(global): {diagnostics.topk_true_outside_ratio_global:.4f}",
    )
    log_print(
        logger,
        f"    最大CC外预测概率质量占比(mean): {diagnostics.outside_prob_mass_ratio_mean:.4f}",
    )
    log_print(
        logger,
        f"    最大CC外预测概率质量占比(global): {diagnostics.outside_prob_mass_ratio_global:.4f}",
    )


def main() -> None:
    config = build_experiment_config()
    setup_seed(SEED)

    data_name = DATASET_NAMES[config.dataset_idx]
    logger = setup_training_logger(log_name=f"{data_name}_gatv2_candidate_domain_log.txt")

    log_print(logger, "=" * 76)
    log_print(logger, "[*] 实验: 单次训练 + 双候选域评测")
    log_print(logger, f"[*] 数据集: {data_name}")
    log_print(logger, f"[*] 设备: {DEVICE}")
    log_print(logger, f"[*] 仅最终快照: {config.final_only}")
    log_print(logger, f"[*] Batch Size: {config.batch_size}")
    log_print(logger, f"[*] hidden_dim={config.hidden_dim}, num_layers={config.num_layers}, dropout={config.dropout}")
    log_print(
        logger,
        f"[*] GFRRLoss: pos_weight={config.pos_weight}, "
        f"lambda_rank={config.lambda_rank}, margin={config.margin}",
    )
    log_print(logger, "=" * 76)

    adj, influ = load_raw_data(DATASETS[config.dataset_idx])
    engineer = FeatureEngineerGFRR(adj)
    cache_name = f"{data_name}_gatv2_candomain_{'final' if config.final_only else 'allsnap'}"
    dataset = engineer.generate_dataset(
        influ,
        cache_name=cache_name,
        use_gfrr=False,
        use_max_cc_graph=False,
        final_only=config.final_only,
        require_source_in_graph=False,
    )

    coverage_stats = summarize_max_cc_coverage(dataset)
    log_print(logger, f"[*] 样本数: {coverage_stats['num_samples']}")
    log_print(
        logger,
        f"[*] 最大CC完整覆盖源点的样本比例: {coverage_stats['covered_samples']}/"
        f"{coverage_stats['num_samples']} ({coverage_stats['coverage_ratio']:.4f})",
    )
    log_print(logger, f"[*] 平均最大CC占感染节点比例: {coverage_stats['avg_max_cc_ratio']:.4f}")

    train_set, val_set, test_set = split_dataset_by_cascade(
        dataset,
        train_ratio=TRAIN_RATIO,
        val_ratio=VAL_RATIO,
    )
    if not train_set or not val_set or not test_set:
        raise RuntimeError("Train/Val/Test 至少有一个为空，请检查数据划分。")

    log_print(
        logger,
        f"[*] 数据划分: Train={len(train_set)}, Val={len(val_set)}, Test={len(test_set)}",
    )

    log_print(logger, "[*] 预计算最短路径矩阵...")
    dist_matrix = precompute_shortest_paths(adj)

    model, training_result = train_model_once(
        train_set=train_set,
        val_set=val_set,
        config=config,
        data_name=data_name,
        dist_matrix=dist_matrix,
        logger=logger,
    )

    val_loader = build_loader(val_set, batch_size=1, shuffle=False)
    test_loader = build_loader(test_set, batch_size=1, shuffle=False)
    eval_results: Dict[str, CandidateEvalResult] = {}
    for candidate_mode in CANDIDATE_MODES:
        candidate_threshold, candidate_val_f1 = find_optimal_threshold(
            model=model,
            loader=val_loader,
            device=DEVICE,
            candidate_mode=candidate_mode,
        )
        log_print(
            logger,
            f"[*] Val最优阈值({candidate_mode}): Threshold={candidate_threshold:.3f}, "
            f"F1={candidate_val_f1:.4f}",
        )
        candidate_stats = summarize_candidate_distribution(test_set, candidate_mode)
        log_print(logger, f"[*] Test候选节点统计({candidate_mode}): {candidate_stats}")
        metrics = evaluate_candidate_mode(
            model=model,
            loader=test_loader,
            device=DEVICE,
            threshold=candidate_threshold,
            recall_k_values=config.recall_k_values,
            dist_matrix=dist_matrix,
            candidate_mode=candidate_mode,
        )
        log_print(logger, f"[*] Test Metrics ({candidate_mode}): {metrics}")
        eval_results[candidate_mode] = CandidateEvalResult(
            candidate_mode=candidate_mode,
            metrics=metrics,
            threshold=candidate_threshold,
            val_f1=candidate_val_f1,
        )

    diagnostics = compute_max_cc_diagnostics(model, test_loader, DEVICE)
    print_comparison_summary(
        logger=logger,
        data_name=data_name,
        eval_results=eval_results,
        diagnostics=diagnostics,
        training_result=training_result,
        recall_k_values=config.recall_k_values,
    )


if __name__ == "__main__":
    main()
