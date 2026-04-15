#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
最大 CC 纯输入诊断脚本

目标:
    1. 只基于原始输入数据分析最大感染连通分量（最大 CC）
    2. 不依赖任何特征工程缓存或模型中间表示
    3. 直接回答“在最大 CC 内，哪些原始结构性质更像源点”
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import networkx as nx
import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components, shortest_path
from scipy.stats import rankdata

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data_loader import load_raw_data  # noqa: E402


DEFAULT_DATASET = "twitter"
DEFAULT_SPLIT = "all"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "analysis_outputs" / "maxcc_pure_input"
DEFAULT_SEED = 3407
DEFAULT_TRAIN_RATIO = 0.70
DEFAULT_VAL_RATIO = 0.10
DATASET_FILE_MAP = {
    "android": "android_25c.SG",
    "christianity": "christianity_25c.SG",
    "douban": "douban_25c.SG",
    "twitter": "twitter_25c.SG",
}

MAXCC_SIZE_BUCKETS = [
    ("<=10", lambda value: value <= 10),
    ("11-20", lambda value: 11 <= value <= 20),
    ("21-50", lambda value: 21 <= value <= 50),
    ("51-100", lambda value: 51 <= value <= 100),
    (">100", lambda value: value > 100),
]

SOURCE_COUNT_BUCKETS = [
    ("1", lambda value: value == 1),
    ("2-3", lambda value: 2 <= value <= 3),
    ("4-6", lambda value: 4 <= value <= 6),
    ("7+", lambda value: value >= 7),
]

SOURCE_DENSITY_BUCKETS = [
    ("<=0.10", lambda value: value <= 0.10),
    ("0.10-0.20", lambda value: 0.10 < value <= 0.20),
    ("0.20-0.35", lambda value: 0.20 < value <= 0.35),
    (">0.35", lambda value: value > 0.35),
]

SOURCE_SPREAD_BUCKETS = [
    ("1", lambda value: value == 1),
    ("2", lambda value: value == 2),
    ("3+", lambda value: value >= 3),
]

SOURCE_PATTERN_CLUSTER_DIRECT_RATIO_THRESHOLD = 0.40
SOURCE_PATTERN_CLUSTER_PROXIMITY_RATIO_THRESHOLD = 0.80
SOURCE_PATTERN_MULTI_BLOCK_MIN_COMPONENTS = 2
SOURCE_PATTERN_CHAIN_MIN_MAX_DISTANCE = 3.0
SOURCE_PATTERN_LABELS = ["簇", "singleton", "多区块", "链", "松散团"]
SOURCE_PATTERN_MODES = {
    "2hop": {
        "display_name": "2-hop 邻近",
        "proximity_hop_threshold": 2,
    },
    "1hop": {
        "display_name": "1-hop 邻近",
        "proximity_hop_threshold": 1,
    },
}

METRIC_SPECS = [
    {
        "name": "global_degree",
        "display_name": "全局度",
        "higher_is_better": True,
        "description": "节点在底层社交图中的全局度数",
    },
    {
        "name": "infected_degree_in_maxcc",
        "display_name": "最大CC内部感染度",
        "higher_is_better": True,
        "description": "节点在最大CC诱导子图内的度数",
    },
    {
        "name": "internal_degree_ratio",
        "display_name": "内部连接占比",
        "higher_is_better": True,
        "description": "最大CC内部度数 / 全局度数",
    },
    {
        "name": "kcore_in_maxcc",
        "display_name": "最大CC内k-core",
        "higher_is_better": True,
        "description": "节点在最大CC诱导子图上的core number",
    },
    {
        "name": "closeness_in_maxcc",
        "display_name": "最大CC内接近中心性",
        "higher_is_better": True,
        "description": "基于最大CC最短路计算的closeness",
    },
    {
        "name": "harmonic_in_maxcc",
        "display_name": "最大CC内harmonic中心性",
        "higher_is_better": True,
        "description": "基于最大CC最短路计算的harmonic centrality",
    },
    {
        "name": "eccentricity_score_in_maxcc",
        "display_name": "最大CC内偏心率反分数",
        "higher_is_better": True,
        "description": "1 - eccentricity / max_eccentricity，越大越居中",
    },
    {
        "name": "boundary_depth_in_maxcc",
        "display_name": "最大CC边界深度",
        "higher_is_better": True,
        "description": "节点到感染边界的归一化最短距离，越大越靠内核",
    },
    {
        "name": "outward_ratio_in_maxcc",
        "display_name": "最大CC内外向比",
        "higher_is_better": True,
        "description": "感染邻居中全局度数低于自己的比例",
    },
    {
        "name": "kinf_vs_best_neighbor_ratio",
        "display_name": "相对最强邻居比",
        "higher_is_better": True,
        "description": "内部感染度 / 邻居中最大内部感染度",
    },
    {
        "name": "stronger_neighbor_ratio",
        "display_name": "更强邻居占比",
        "higher_is_better": False,
        "description": "内部感染邻居中度数比自己更大的比例，越低越像源点",
    },
    {
        "name": "avg_distance_in_maxcc",
        "display_name": "最大CC内平均距离",
        "higher_is_better": False,
        "description": "节点到最大CC其他节点的平均最短路，越低越居中",
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="最大 CC 纯输入诊断脚本")
    parser.add_argument("--dataset", default=DEFAULT_DATASET, choices=sorted(DATASET_FILE_MAP.keys()), help="数据集名称")
    parser.add_argument(
        "--split",
        default=DEFAULT_SPLIT,
        choices=["train", "val", "test", "all"],
        help="分析使用的数据划分；all 表示使用全部级联",
    )
    parser.add_argument("--output-dir", default=None, help="输出目录")
    parser.add_argument("--train-ratio", type=float, default=DEFAULT_TRAIN_RATIO, help="train 划分比例")
    parser.add_argument("--val-ratio", type=float, default=DEFAULT_VAL_RATIO, help="val 划分比例")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="划分随机种子")
    return parser.parse_args()


def ensure_directory(directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)


def write_json(file_path: Path, payload: object) -> None:
    ensure_directory(file_path.parent)
    with file_path.open("w", encoding="utf-8") as file_obj:
        json.dump(payload, file_obj, ensure_ascii=False, indent=2)


def write_csv(file_path: Path, rows: Sequence[Dict[str, object]]) -> None:
    ensure_directory(file_path.parent)
    if not rows:
        file_path.write_text("", encoding="utf-8")
        return

    field_names: List[str] = []
    seen = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                field_names.append(key)

    with file_path.open("w", encoding="utf-8", newline="") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=field_names)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def mean_or_zero(values: Iterable[float]) -> float:
    values = list(values)
    if not values:
        return 0.0
    return float(np.mean(np.asarray(values, dtype=np.float64)))


def split_cascade_indices(
    num_cascades: int,
    split_name: str,
    seed: int,
    train_ratio: float,
    val_ratio: float,
) -> np.ndarray:
    if split_name == "all":
        return np.arange(num_cascades, dtype=np.int64)

    if train_ratio <= 0 or val_ratio < 0 or train_ratio + val_ratio >= 1:
        raise ValueError("train_ratio 和 val_ratio 必须满足: train_ratio > 0, val_ratio >= 0, train+val < 1")

    cascade_ids = np.arange(num_cascades, dtype=np.int64)
    rng = np.random.RandomState(seed)
    rng.shuffle(cascade_ids)

    train_end = int(num_cascades * train_ratio)
    val_end = int(num_cascades * (train_ratio + val_ratio))
    split_map = {
        "train": cascade_ids[:train_end],
        "val": cascade_ids[train_end:val_end],
        "test": cascade_ids[val_end:],
    }
    return split_map[split_name]


def build_neighbors_list(adjacency_matrix: sp.csr_matrix) -> List[np.ndarray]:
    return [
        adjacency_matrix.indices[adjacency_matrix.indptr[node_id]:adjacency_matrix.indptr[node_id + 1]]
        for node_id in range(adjacency_matrix.shape[0])
    ]


def extract_max_cc_nodes(adjacency_matrix: sp.csr_matrix, infected_node_ids: np.ndarray) -> np.ndarray:
    infected_node_ids = np.asarray(infected_node_ids, dtype=np.int64)
    if infected_node_ids.size == 0:
        return np.array([], dtype=np.int64)
    if infected_node_ids.size == 1:
        return infected_node_ids.copy()

    infected_subgraph = adjacency_matrix[infected_node_ids][:, infected_node_ids]
    num_components, labels = connected_components(infected_subgraph, directed=False, return_labels=True)
    if num_components <= 0:
        return np.array([], dtype=np.int64)

    component_sizes = np.bincount(labels)
    max_component_id = int(np.argmax(component_sizes))
    return infected_node_ids[labels == max_component_id]


def compute_source_spread(labels: np.ndarray, infected_node_ids: np.ndarray, source_node_ids: np.ndarray) -> int:
    infected_to_local = {int(node_id): local_idx for local_idx, node_id in enumerate(infected_node_ids.tolist())}
    source_component_ids = set()
    for node_id in source_node_ids.tolist():
        local_idx = infected_to_local.get(int(node_id))
        if local_idx is not None:
            source_component_ids.add(int(labels[local_idx]))
    return len(source_component_ids)


def build_local_graph_metrics(
    adjacency_matrix: sp.csr_matrix,
    neighbors_list: Sequence[np.ndarray],
    global_degrees: np.ndarray,
    infected_set: set,
    max_cc_node_ids: np.ndarray,
) -> Dict[str, np.ndarray]:
    max_cc_node_ids = np.asarray(max_cc_node_ids, dtype=np.int64)
    node_count = int(max_cc_node_ids.size)
    if node_count == 0:
        return {metric_spec["name"]: np.zeros(0, dtype=np.float32) for metric_spec in METRIC_SPECS}

    local_adj = adjacency_matrix[max_cc_node_ids][:, max_cc_node_ids].tocsr()
    local_degrees = np.asarray(local_adj.getnnz(axis=1)).astype(np.float32)
    global_degree_values = global_degrees[max_cc_node_ids].astype(np.float32)
    internal_degree_ratio = local_degrees / np.maximum(global_degree_values, 1.0)

    if node_count == 1:
        return {
            "global_degree": global_degree_values,
            "infected_degree_in_maxcc": local_degrees,
            "internal_degree_ratio": internal_degree_ratio,
            "kcore_in_maxcc": np.zeros(1, dtype=np.float32),
            "closeness_in_maxcc": np.ones(1, dtype=np.float32),
            "harmonic_in_maxcc": np.ones(1, dtype=np.float32),
            "eccentricity_score_in_maxcc": np.ones(1, dtype=np.float32),
            "boundary_depth_in_maxcc": np.zeros(1, dtype=np.float32),
            "outward_ratio_in_maxcc": np.zeros(1, dtype=np.float32),
            "kinf_vs_best_neighbor_ratio": np.ones(1, dtype=np.float32),
            "stronger_neighbor_ratio": np.zeros(1, dtype=np.float32),
            "avg_distance_in_maxcc": np.zeros(1, dtype=np.float32),
        }

    dist_matrix = shortest_path(local_adj, directed=False, unweighted=True, return_predecessors=False)
    dist_matrix = np.asarray(dist_matrix, dtype=np.float32)

    finite_matrix = np.where(np.isfinite(dist_matrix), dist_matrix, 0.0)
    sum_dist = finite_matrix.sum(axis=1)
    avg_distance = (sum_dist / max(node_count - 1, 1)).astype(np.float32)
    closeness = np.divide(
        max(node_count - 1, 1),
        np.maximum(sum_dist, 1e-8),
        out=np.zeros(node_count, dtype=np.float32),
        where=sum_dist > 0,
    ).astype(np.float32)

    reciprocal = np.zeros_like(dist_matrix, dtype=np.float32)
    valid_mask = dist_matrix > 0
    reciprocal[valid_mask] = 1.0 / dist_matrix[valid_mask]
    harmonic = (reciprocal.sum(axis=1) / max(node_count - 1, 1)).astype(np.float32)

    eccentricity = dist_matrix.max(axis=1)
    max_eccentricity = max(float(eccentricity.max()) if eccentricity.size > 0 else 0.0, 1.0)
    eccentricity_score = (1.0 - (eccentricity / max_eccentricity)).astype(np.float32)

    if hasattr(nx, "from_scipy_sparse_array"):
        graph_obj = nx.from_scipy_sparse_array(local_adj)
    else:
        graph_obj = nx.from_scipy_sparse_matrix(local_adj)
    try:
        kcore_values_dict = nx.core_number(graph_obj)
        kcore_values = np.asarray(
            [float(kcore_values_dict.get(local_idx, 0.0)) for local_idx in range(node_count)],
            dtype=np.float32,
        )
    except Exception:
        kcore_values = np.zeros(node_count, dtype=np.float32)

    boundary_local_indices = []
    for local_idx, node_id in enumerate(max_cc_node_ids.tolist()):
        has_outside_neighbor = any(int(neighbor_id) not in infected_set for neighbor_id in neighbors_list[int(node_id)])
        if has_outside_neighbor:
            boundary_local_indices.append(local_idx)
    if not boundary_local_indices:
        boundary_local_indices = list(range(node_count))

    boundary_dist = dist_matrix[:, boundary_local_indices].min(axis=1).astype(np.float32)
    max_boundary_dist = max(float(boundary_dist.max()) if boundary_dist.size > 0 else 0.0, 1.0)
    boundary_depth = (boundary_dist / max_boundary_dist).astype(np.float32)

    outward_ratio = np.zeros(node_count, dtype=np.float32)
    stronger_neighbor_ratio = np.zeros(node_count, dtype=np.float32)
    kinf_vs_best_neighbor_ratio = np.ones(node_count, dtype=np.float32)
    for local_idx in range(node_count):
        start = local_adj.indptr[local_idx]
        end = local_adj.indptr[local_idx + 1]
        local_neighbors = local_adj.indices[start:end]
        if local_neighbors.size == 0:
            continue

        my_global_degree = global_degree_values[local_idx]
        neighbor_global_degrees = global_degree_values[local_neighbors]
        outward_ratio[local_idx] = float(np.mean(neighbor_global_degrees < my_global_degree))

        neighbor_local_degrees = local_degrees[local_neighbors]
        stronger_neighbor_ratio[local_idx] = float(np.mean(neighbor_local_degrees > local_degrees[local_idx]))
        best_neighbor_degree = float(neighbor_local_degrees.max()) if neighbor_local_degrees.size > 0 else 0.0
        if best_neighbor_degree > 0:
            kinf_vs_best_neighbor_ratio[local_idx] = float(local_degrees[local_idx] / best_neighbor_degree)

    return {
        "global_degree": global_degree_values,
        "infected_degree_in_maxcc": local_degrees,
        "internal_degree_ratio": internal_degree_ratio.astype(np.float32),
        "kcore_in_maxcc": kcore_values,
        "closeness_in_maxcc": closeness,
        "harmonic_in_maxcc": harmonic,
        "eccentricity_score_in_maxcc": eccentricity_score,
        "boundary_depth_in_maxcc": boundary_depth,
        "outward_ratio_in_maxcc": outward_ratio,
        "kinf_vs_best_neighbor_ratio": kinf_vs_best_neighbor_ratio.astype(np.float32),
        "stronger_neighbor_ratio": stronger_neighbor_ratio,
        "avg_distance_in_maxcc": avg_distance,
    }


def compute_source_pair_stats(
    local_adj: sp.csr_matrix,
    source_local_indices: np.ndarray,
    proximity_hop_threshold: int,
) -> Dict[str, float]:
    source_local_indices = np.asarray(source_local_indices, dtype=np.int64)
    if proximity_hop_threshold < 1:
        raise ValueError("proximity_hop_threshold 必须 >= 1")

    if source_local_indices.size <= 1:
        return {
            "mean_source_pair_distance": 0.0,
            "max_source_pair_distance": 0.0,
            "direct_neighbor_ratio": 0.0,
            "within_proximity_ratio": 0.0,
            "source_proximity_component_count": float(source_local_indices.size),
        }

    pairwise_dist = shortest_path(
        local_adj,
        directed=False,
        unweighted=True,
        indices=source_local_indices,
        return_predecessors=False,
    )
    pairwise_dist = np.asarray(pairwise_dist, dtype=np.float32)
    source_to_source = pairwise_dist[:, source_local_indices]
    upper_mask = np.triu(np.ones_like(source_to_source, dtype=bool), k=1)
    valid_distances = source_to_source[upper_mask]
    valid_distances = valid_distances[np.isfinite(valid_distances)]

    if valid_distances.size == 0:
        return {
            "mean_source_pair_distance": 0.0,
            "max_source_pair_distance": 0.0,
            "direct_neighbor_ratio": 0.0,
            "within_proximity_ratio": 0.0,
            "source_proximity_component_count": float(source_local_indices.size),
        }

    proximity_graph = sp.csr_matrix((pairwise_dist[:, source_local_indices] <= proximity_hop_threshold).astype(np.int8))
    proximity_graph.setdiag(0)
    proximity_components, _ = connected_components(proximity_graph, directed=False, return_labels=True)

    return {
        "mean_source_pair_distance": float(valid_distances.mean()),
        "max_source_pair_distance": float(valid_distances.max()),
        "direct_neighbor_ratio": float(np.mean(valid_distances == 1)),
        "within_proximity_ratio": float(np.mean(valid_distances <= proximity_hop_threshold)),
        "source_proximity_component_count": float(proximity_components),
    }


def classify_source_pattern(
    source_count_in_maxcc: int,
    direct_neighbor_ratio: float,
    within_proximity_ratio: float,
    proximity_component_count: float,
    max_source_pair_distance: float,
) -> str:
    if source_count_in_maxcc <= 1:
        return "singleton"
    if proximity_component_count >= SOURCE_PATTERN_MULTI_BLOCK_MIN_COMPONENTS:
        return "多区块"
    if (
        direct_neighbor_ratio >= SOURCE_PATTERN_CLUSTER_DIRECT_RATIO_THRESHOLD
        or within_proximity_ratio >= SOURCE_PATTERN_CLUSTER_PROXIMITY_RATIO_THRESHOLD
    ):
        return "簇"
    if max_source_pair_distance >= SOURCE_PATTERN_CHAIN_MIN_MAX_DISTANCE:
        return "链"
    return "松散团"


def orient_scores(values: np.ndarray, higher_is_better: bool) -> np.ndarray:
    return values.astype(np.float32) if higher_is_better else (-values).astype(np.float32)


def compute_average_precision(labels: np.ndarray, scores: np.ndarray) -> float:
    positives = int(labels.sum())
    if positives <= 0:
        return 0.0
    if scores.size == 0:
        return 0.0
    if np.allclose(scores, scores[0]):
        return float(labels.mean())

    order = np.argsort(-scores, kind="mergesort")
    ranked_labels = labels[order].astype(np.int64)
    cum_tp = np.cumsum(ranked_labels)
    positive_positions = np.where(ranked_labels == 1)[0]
    if positive_positions.size == 0:
        return 0.0
    precision_at_hits = cum_tp[positive_positions] / (positive_positions + 1)
    return float(np.mean(precision_at_hits))


def compute_pairwise_auc(source_scores: np.ndarray, non_source_scores: np.ndarray) -> float:
    if source_scores.size == 0 or non_source_scores.size == 0:
        return 0.0
    pairwise_diff = source_scores[:, None] - non_source_scores[None, :]
    win = np.mean(pairwise_diff > 0)
    tie = np.mean(pairwise_diff == 0)
    return float(win + 0.5 * tie)


def compute_expected_topk_recall(labels: np.ndarray, scores: np.ndarray, topk: int) -> float:
    if topk <= 0 or labels.size == 0:
        return 0.0
    if topk >= labels.size:
        return float(labels.mean())

    order = np.argsort(-scores, kind="mergesort")
    kth_score = float(scores[order[topk - 1]])
    above_mask = scores > kth_score
    tie_mask = np.isclose(scores, kth_score)

    expected_hits = float(labels[above_mask].sum())
    fixed_count = int(above_mask.sum())
    remaining_slots = max(topk - fixed_count, 0)

    tie_count = int(tie_mask.sum())
    if remaining_slots > 0 and tie_count > 0:
        expected_hits += remaining_slots * float(labels[tie_mask].sum()) / tie_count

    return expected_hits / max(topk, 1)


def summarize_metric_rankings(cascade_metric_rows: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    metric_rows: List[Dict[str, object]] = []
    for metric_spec in METRIC_SPECS:
        metric_name = metric_spec["name"]
        selected_rows = [row for row in cascade_metric_rows if row["metric"] == metric_name]
        if not selected_rows:
            continue

        metric_rows.append(
            {
                "metric": metric_name,
                "display_name": metric_spec["display_name"],
                "direction": "higher_is_source_like" if metric_spec["higher_is_better"] else "lower_is_source_like",
                "description": metric_spec["description"],
                "valid_cascade_count": len(selected_rows),
                "mean_source_percentile": round(mean_or_zero(row["mean_source_percentile"] for row in selected_rows), 6),
                "top1_is_source_rate": round(mean_or_zero(row["top1_is_source"] for row in selected_rows), 6),
                "single_source_top1_acc": round(
                    mean_or_zero(
                        row["single_source_top1_hit"]
                        for row in selected_rows
                        if row["single_source_top1_hit"] is not None
                    ),
                    6,
                ),
                "topk_true_recall": round(mean_or_zero(row["topk_true_recall"] for row in selected_rows), 6),
                "mean_ap": round(mean_or_zero(row["average_precision"] for row in selected_rows), 6),
                "mean_pairwise_auc": round(mean_or_zero(row["pairwise_auc"] for row in selected_rows), 6),
                "source_mean_raw": round(mean_or_zero(row["source_mean_raw"] for row in selected_rows), 6),
                "non_source_mean_raw": round(mean_or_zero(row["non_source_mean_raw"] for row in selected_rows), 6),
                "directional_gap_mean": round(mean_or_zero(row["directional_gap"] for row in selected_rows), 6),
            }
        )

    metric_rows.sort(
        key=lambda row: (
            float(row["mean_ap"]),
            float(row["mean_pairwise_auc"]),
            float(row["mean_source_percentile"]),
        ),
        reverse=True,
    )
    return metric_rows


def bucketize(rows: Sequence[Dict[str, object]], field_name: str, bucket_defs) -> List[Dict[str, object]]:
    bucket_rows = []
    for bucket_name, predicate in bucket_defs:
        selected_rows = [
            row for row in rows
            if row.get(field_name) is not None and predicate(float(row[field_name]))
        ]
        if not selected_rows:
            continue

        bucket_rows.append(
            {
                "bucket": bucket_name,
                "count": len(selected_rows),
                "infected_size_mean": round(mean_or_zero(row["infected_size"] for row in selected_rows), 6),
                "max_cc_size_mean": round(mean_or_zero(row["max_cc_size"] for row in selected_rows), 6),
                "source_count_mean": round(mean_or_zero(row["source_count"] for row in selected_rows), 6),
                "source_in_maxcc_mean": round(mean_or_zero(row["source_in_maxcc_count"] for row in selected_rows), 6),
                "source_density_mean": round(mean_or_zero(row["source_density_in_maxcc"] for row in selected_rows), 6),
                "all_sources_in_maxcc_ratio": round(mean_or_zero(row["all_sources_in_maxcc"] for row in selected_rows), 6),
                "source_boundary_depth_mean": round(mean_or_zero(row["source_boundary_depth_mean"] for row in selected_rows), 6),
                "mean_source_pair_distance": round(mean_or_zero(row["mean_source_pair_distance"] for row in selected_rows), 6),
            }
        )
    return bucket_rows


def summarize_cascade_rows(cascade_rows: Sequence[Dict[str, object]]) -> Dict[str, object]:
    if not cascade_rows:
        return {}
    return {
        "cascade_count": len(cascade_rows),
        "infected_size_mean": round(mean_or_zero(row["infected_size"] for row in cascade_rows), 6),
        "source_count_mean": round(mean_or_zero(row["source_count"] for row in cascade_rows), 6),
        "max_cc_size_mean": round(mean_or_zero(row["max_cc_size"] for row in cascade_rows), 6),
        "max_cc_ratio_mean": round(mean_or_zero(row["max_cc_ratio"] for row in cascade_rows), 6),
        "source_in_maxcc_mean": round(mean_or_zero(row["source_in_maxcc_count"] for row in cascade_rows), 6),
        "source_density_in_maxcc_mean": round(mean_or_zero(row["source_density_in_maxcc"] for row in cascade_rows), 6),
        "all_sources_in_maxcc_ratio": round(mean_or_zero(row["all_sources_in_maxcc"] for row in cascade_rows), 6),
        "has_source_in_maxcc_ratio": round(mean_or_zero(row["source_in_maxcc_count"] > 0 for row in cascade_rows), 6),
        "source_spread_mean": round(mean_or_zero(row["source_spread_over_infected_cc"] for row in cascade_rows), 6),
        "mean_source_pair_distance": round(mean_or_zero(row["mean_source_pair_distance"] for row in cascade_rows), 6),
        "direct_neighbor_ratio_mean": round(mean_or_zero(row["direct_neighbor_ratio"] for row in cascade_rows), 6),
        "within_two_hop_ratio_mean": round(mean_or_zero(row["within_two_hop_ratio"] for row in cascade_rows), 6),
        "source_boundary_depth_mean": round(mean_or_zero(row["source_boundary_depth_mean"] for row in cascade_rows), 6),
    }


def summarize_source_patterns(
    cascade_rows: Sequence[Dict[str, object]],
    pattern_field_name: str,
) -> List[Dict[str, object]]:
    if not cascade_rows:
        return []

    total_count = len(cascade_rows)
    pattern_rows: List[Dict[str, object]] = []
    for pattern_name in SOURCE_PATTERN_LABELS:
        selected_rows = [row for row in cascade_rows if str(row[pattern_field_name]) == pattern_name]
        pattern_rows.append(
            {
                "source_pattern": pattern_name,
                "count": len(selected_rows),
                "ratio": round(float(len(selected_rows) / total_count), 6),
                "max_cc_size_mean": round(mean_or_zero(row["max_cc_size"] for row in selected_rows), 6),
                "source_count_mean": round(mean_or_zero(row["source_count"] for row in selected_rows), 6),
                "source_density_mean": round(mean_or_zero(row["source_density_in_maxcc"] for row in selected_rows), 6),
                "mean_source_pair_distance": round(mean_or_zero(row["mean_source_pair_distance"] for row in selected_rows), 6),
                "source_boundary_depth_mean": round(mean_or_zero(row["source_boundary_depth_mean"] for row in selected_rows), 6),
            }
        )
    return pattern_rows


def format_report_text(
    dataset_name: str,
    split_name: str,
    overall_summary: Dict[str, object],
    source_pattern_rows_2hop: Sequence[Dict[str, object]],
    source_pattern_rows_1hop: Sequence[Dict[str, object]],
    metric_rows: Sequence[Dict[str, object]],
    bucket_tables: Dict[str, List[Dict[str, object]]],
) -> str:
    lines = []
    lines.append("=" * 76)
    lines.append(f"最大 CC 纯输入诊断报告 - {dataset_name} ({split_name})")
    lines.append("=" * 76)
    lines.append("")
    lines.append("[最大 CC 总体概况]")
    for key, value in overall_summary.items():
        lines.append(f"  - {key}: {value}")
    lines.append("")
    lines.append("[源点形态分布]")
    for mode_name, pattern_rows in (
        (SOURCE_PATTERN_MODES["2hop"]["display_name"], source_pattern_rows_2hop),
        (SOURCE_PATTERN_MODES["1hop"]["display_name"], source_pattern_rows_1hop),
    ):
        lines.append(f"  - {mode_name}")
        if not pattern_rows:
            lines.append("    * 当前划分下无可用样本")
            continue

        for row in pattern_rows:
            lines.append(
                "    * "
                f"{row['source_pattern']}: ratio={row['ratio']:.4f}, "
                f"maxcc={row['max_cc_size_mean']:.2f}, src={row['source_count_mean']:.2f}, "
                f"density={row['source_density_mean']:.4f}, pair_dist={row['mean_source_pair_distance']:.2f}, "
                f"boundary_depth={row['source_boundary_depth_mean']:.4f}"
            )
    lines.append("")
    lines.append("[纯结构指标排序 Top 12]")
    if not metric_rows:
        lines.append("  - 当前划分下无可比较的最大 CC 样本")
    else:
        for row in metric_rows[: min(12, len(metric_rows))]:
            lines.append(
                "  - "
                f"{row['display_name']} ({row['metric']}): "
                f"AP={row['mean_ap']:.4f}, AUC={row['mean_pairwise_auc']:.4f}, "
                f"src_pct={row['mean_source_percentile']:.4f}, top1={row['top1_is_source_rate']:.4f}, "
                f"topk={row['topk_true_recall']:.4f}, single={row['single_source_top1_acc']:.4f}"
            )
    lines.append("")
    lines.append("[难度分层]")
    for table_name, table_rows in bucket_tables.items():
        lines.append(f"  - {table_name}")
        for row in table_rows:
            lines.append(
                "    * "
                f"{row['bucket']}: n={row['count']}, infected={row['infected_size_mean']:.2f}, "
                f"maxcc={row['max_cc_size_mean']:.2f}, src={row['source_count_mean']:.2f}, "
                f"src_in_maxcc={row['source_in_maxcc_mean']:.2f}, density={row['source_density_mean']:.4f}, "
                f"all_in_maxcc={row['all_sources_in_maxcc_ratio']:.4f}, "
                f"boundary_depth={row['source_boundary_depth_mean']:.4f}, "
                f"pair_dist={row['mean_source_pair_distance']:.2f}"
            )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    sg_file_name = DATASET_FILE_MAP[args.dataset]
    output_dir = Path(args.output_dir) if args.output_dir else DEFAULT_OUTPUT_ROOT / args.dataset
    ensure_directory(output_dir)

    adjacency_matrix, influence_matrices = load_raw_data(sg_file_name)
    adjacency_matrix = adjacency_matrix.tocsr() if sp.issparse(adjacency_matrix) else sp.csr_matrix(adjacency_matrix)
    if influence_matrices is None or len(influence_matrices) == 0:
        raise ValueError(f"{sg_file_name} 中没有 influ_mat_list")

    cascade_indices = split_cascade_indices(
        num_cascades=int(len(influence_matrices)),
        split_name=args.split,
        seed=args.seed,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
    )

    neighbors_list = build_neighbors_list(adjacency_matrix)
    global_degrees = np.asarray(adjacency_matrix.getnnz(axis=1)).astype(np.float32)

    cascade_rows: List[Dict[str, object]] = []
    cascade_metric_rows: List[Dict[str, object]] = []

    print("=" * 76)
    print(f"[*] 最大 CC 纯输入诊断 - {args.dataset}")
    print(f"[*] 使用级联数: {len(cascade_indices)} / {len(influence_matrices)}")
    print(f"[*] 输出目录: {output_dir}")
    print("=" * 76)

    for running_index, cascade_id in enumerate(cascade_indices.tolist(), start=1):
        influ_mat = influence_matrices[int(cascade_id)]
        source_node_ids = np.where(influ_mat[:, 0] == 1)[0].astype(np.int64)
        infected_node_ids = np.where(influ_mat[:, -1] == 1)[0].astype(np.int64)

        if infected_node_ids.size == 0 or source_node_ids.size == 0:
            continue

        infected_subgraph = adjacency_matrix[infected_node_ids][:, infected_node_ids]
        num_components, component_labels = connected_components(infected_subgraph, directed=False, return_labels=True)
        max_cc_node_ids = extract_max_cc_nodes(adjacency_matrix, infected_node_ids)
        if max_cc_node_ids.size == 0:
            continue

        infected_set = set(int(node_id) for node_id in infected_node_ids.tolist())
        source_in_maxcc_mask = np.isin(source_node_ids, max_cc_node_ids)
        source_in_maxcc_count = int(source_in_maxcc_mask.sum())
        source_local_indices = np.where(np.isin(max_cc_node_ids, source_node_ids))[0].astype(np.int64)

        metric_map = build_local_graph_metrics(
            adjacency_matrix=adjacency_matrix,
            neighbors_list=neighbors_list,
            global_degrees=global_degrees,
            infected_set=infected_set,
            max_cc_node_ids=max_cc_node_ids,
        )
        local_adj = adjacency_matrix[max_cc_node_ids][:, max_cc_node_ids].tocsr()
        pair_stats_2hop = compute_source_pair_stats(
            local_adj,
            source_local_indices,
            proximity_hop_threshold=SOURCE_PATTERN_MODES["2hop"]["proximity_hop_threshold"],
        )
        source_pattern_2hop = classify_source_pattern(
            source_count_in_maxcc=source_in_maxcc_count,
            direct_neighbor_ratio=pair_stats_2hop["direct_neighbor_ratio"],
            within_proximity_ratio=pair_stats_2hop["within_proximity_ratio"],
            proximity_component_count=pair_stats_2hop["source_proximity_component_count"],
            max_source_pair_distance=pair_stats_2hop["max_source_pair_distance"],
        )
        pair_stats_1hop = compute_source_pair_stats(
            local_adj,
            source_local_indices,
            proximity_hop_threshold=SOURCE_PATTERN_MODES["1hop"]["proximity_hop_threshold"],
        )
        source_pattern_1hop = classify_source_pattern(
            source_count_in_maxcc=source_in_maxcc_count,
            direct_neighbor_ratio=pair_stats_1hop["direct_neighbor_ratio"],
            within_proximity_ratio=pair_stats_1hop["within_proximity_ratio"],
            proximity_component_count=pair_stats_1hop["source_proximity_component_count"],
            max_source_pair_distance=pair_stats_1hop["max_source_pair_distance"],
        )

        source_boundary_depth_mean = 0.0
        if source_local_indices.size > 0:
            source_boundary_depth_mean = float(metric_map["boundary_depth_in_maxcc"][source_local_indices].mean())

        cascade_rows.append(
            {
                "cascade_id": int(cascade_id),
                "infected_size": int(infected_node_ids.size),
                "source_count": int(source_node_ids.size),
                "max_cc_size": int(max_cc_node_ids.size),
                "max_cc_ratio": round(float(max_cc_node_ids.size / max(infected_node_ids.size, 1)), 6),
                "source_in_maxcc_count": source_in_maxcc_count,
                "source_density_in_maxcc": round(float(source_in_maxcc_count / max(max_cc_node_ids.size, 1)), 6),
                "all_sources_in_maxcc": int(source_in_maxcc_count == source_node_ids.size),
                "num_infected_components": int(num_components),
                "source_spread_over_infected_cc": int(
                    compute_source_spread(component_labels, infected_node_ids, source_node_ids)
                ),
                "source_boundary_depth_mean": round(source_boundary_depth_mean, 6),
                "source_pattern": source_pattern_2hop,
                "source_pattern_2hop": source_pattern_2hop,
                "source_pattern_1hop": source_pattern_1hop,
                "mean_source_pair_distance": round(pair_stats_2hop["mean_source_pair_distance"], 6),
                "max_source_pair_distance": round(pair_stats_2hop["max_source_pair_distance"], 6),
                "direct_neighbor_ratio": round(pair_stats_2hop["direct_neighbor_ratio"], 6),
                "within_one_hop_ratio": round(pair_stats_1hop["within_proximity_ratio"], 6),
                "within_two_hop_ratio": round(pair_stats_2hop["within_proximity_ratio"], 6),
                "source_proximity_component_count": round(pair_stats_2hop["source_proximity_component_count"], 6),
                "source_proximity_component_count_2hop": round(pair_stats_2hop["source_proximity_component_count"], 6),
                "source_proximity_component_count_1hop": round(pair_stats_1hop["source_proximity_component_count"], 6),
            }
        )

        if source_in_maxcc_count == 0 or source_in_maxcc_count == max_cc_node_ids.size:
            continue

        source_local_mask = np.isin(max_cc_node_ids, source_node_ids)
        non_source_local_mask = ~source_local_mask

        for metric_spec in METRIC_SPECS:
            metric_name = metric_spec["name"]
            raw_values = metric_map[metric_name].astype(np.float32)
            scores = orient_scores(raw_values, higher_is_better=bool(metric_spec["higher_is_better"]))
            if raw_values.size <= 1:
                continue

            order = np.argsort(-scores, kind="mergesort")
            average_ranks = rankdata(-scores, method="average") - 1.0
            percentiles = 1.0 - (average_ranks / max(raw_values.size - 1, 1))
            topk = source_in_maxcc_count
            topk_true_recall = compute_expected_topk_recall(source_local_mask.astype(np.float32), scores, topk)
            ap = compute_average_precision(source_local_mask.astype(np.int64), scores)
            pairwise_auc = compute_pairwise_auc(scores[source_local_mask], scores[non_source_local_mask])
            top_score = float(scores[order[0]])
            top_mask = np.isclose(scores, top_score)
            top1_is_source = float(source_local_mask[top_mask].mean())

            single_source_top1_hit = None
            if source_in_maxcc_count == 1:
                single_source_top1_hit = float(source_local_mask[top_mask].sum() / max(top_mask.sum(), 1))

            cascade_metric_rows.append(
                {
                    "cascade_id": int(cascade_id),
                    "metric": metric_name,
                    "mean_source_percentile": float(percentiles[source_local_mask].mean()),
                    "top1_is_source": top1_is_source,
                    "single_source_top1_hit": single_source_top1_hit,
                    "topk_true_recall": topk_true_recall,
                    "average_precision": ap,
                    "pairwise_auc": pairwise_auc,
                    "source_mean_raw": float(raw_values[source_local_mask].mean()),
                    "non_source_mean_raw": float(raw_values[non_source_local_mask].mean()),
                    "directional_gap": float(scores[source_local_mask].mean() - scores[non_source_local_mask].mean()),
                }
            )

        if running_index % 200 == 0:
            print(f"    已处理 {running_index}/{len(cascade_indices)} 个级联")

    overall_summary = summarize_cascade_rows(cascade_rows)
    source_pattern_rows_2hop = summarize_source_patterns(cascade_rows, pattern_field_name="source_pattern_2hop")
    source_pattern_rows_1hop = summarize_source_patterns(cascade_rows, pattern_field_name="source_pattern_1hop")
    metric_rows = summarize_metric_rankings(cascade_metric_rows)
    bucket_tables = {
        "max_cc_size": bucketize(cascade_rows, "max_cc_size", MAXCC_SIZE_BUCKETS),
        "source_count": bucketize(cascade_rows, "source_count", SOURCE_COUNT_BUCKETS),
        "source_density": bucketize(cascade_rows, "source_density_in_maxcc", SOURCE_DENSITY_BUCKETS),
        "source_spread": bucketize(cascade_rows, "source_spread_over_infected_cc", SOURCE_SPREAD_BUCKETS),
    }

    summary_payload = {
        "dataset": args.dataset,
        "split": args.split,
        "overall_summary": overall_summary,
        "source_pattern_summary": source_pattern_rows_2hop,
        "source_pattern_summary_2hop": source_pattern_rows_2hop,
        "source_pattern_summary_1hop": source_pattern_rows_1hop,
        "top_metric_ranking": metric_rows[: min(20, len(metric_rows))],
        "metric_count": len(metric_rows),
        "cascade_count": len(cascade_rows),
    }

    write_json(output_dir / "summary.json", summary_payload)
    write_csv(output_dir / "cascade_stats.csv", cascade_rows)
    write_csv(output_dir / "metric_ranking.csv", metric_rows)
    write_csv(output_dir / "cascade_metric_ranking.csv", cascade_metric_rows)
    write_csv(output_dir / "source_pattern_summary.csv", source_pattern_rows_2hop)
    write_csv(output_dir / "source_pattern_summary_2hop.csv", source_pattern_rows_2hop)
    write_csv(output_dir / "source_pattern_summary_1hop.csv", source_pattern_rows_1hop)
    write_csv(output_dir / "bucket_maxcc_size.csv", bucket_tables["max_cc_size"])
    write_csv(output_dir / "bucket_source_count.csv", bucket_tables["source_count"])
    write_csv(output_dir / "bucket_source_density.csv", bucket_tables["source_density"])
    write_csv(output_dir / "bucket_source_spread.csv", bucket_tables["source_spread"])

    report_text = format_report_text(
        dataset_name=args.dataset,
        split_name=args.split,
        overall_summary=overall_summary,
        source_pattern_rows_2hop=source_pattern_rows_2hop,
        source_pattern_rows_1hop=source_pattern_rows_1hop,
        metric_rows=metric_rows,
        bucket_tables=bucket_tables,
    )
    report_path = output_dir / "report.txt"
    report_path.write_text(report_text, encoding="utf-8")
    print(report_text)
    print(f"[*] 输出已写入: {output_dir}")


if __name__ == "__main__":
    main()
