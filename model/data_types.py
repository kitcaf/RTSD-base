from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Dict

import numpy as np
import scipy.sparse as sp
import torch


@dataclass(frozen=True)
class GraphStaticContext:
    adjacency_csr: sp.csr_matrix
    neighbors_list: tuple[np.ndarray, ...]
    degrees: np.ndarray
    degree_percentile: np.ndarray
    max_degree: float
    num_nodes: int


@dataclass(frozen=True)
class ContextGraphSample:
    cascade_id: int
    orig_node_ids: torch.LongTensor
    edge_index_ctx: torch.LongTensor
    x: torch.FloatTensor
    maxcc_mask: torch.BoolTensor
    loss_mask: torch.BoolTensor
    y_root: torch.FloatTensor
    k_label: int
    pair1_index: torch.LongTensor
    pair1_features: torch.FloatTensor
    pair1_compat_label: torch.FloatTensor
    pair1_compat_mask: torch.BoolTensor
    pair2_index: torch.LongTensor
    pair2_features: torch.FloatTensor
    pair2_compat_label: torch.FloatTensor
    pair2_compat_mask: torch.BoolTensor
    maxcc_local_idx: torch.LongTensor
    maxcc_orig_node_ids: torch.LongTensor

    def to(self, device: torch.device | str) -> "ContextGraphSample":
        return replace(
            self,
            orig_node_ids=self.orig_node_ids.to(device),
            edge_index_ctx=self.edge_index_ctx.to(device),
            x=self.x.to(device),
            maxcc_mask=self.maxcc_mask.to(device),
            loss_mask=self.loss_mask.to(device),
            y_root=self.y_root.to(device),
            pair1_index=self.pair1_index.to(device),
            pair1_features=self.pair1_features.to(device),
            pair1_compat_label=self.pair1_compat_label.to(device),
            pair1_compat_mask=self.pair1_compat_mask.to(device),
            pair2_index=self.pair2_index.to(device),
            pair2_features=self.pair2_features.to(device),
            pair2_compat_label=self.pair2_compat_label.to(device),
            pair2_compat_mask=self.pair2_compat_mask.to(device),
            maxcc_local_idx=self.maxcc_local_idx.to(device),
            maxcc_orig_node_ids=self.maxcc_orig_node_ids.to(device),
        )

    @property
    def num_context_nodes(self) -> int:
        return int(self.x.size(0))

    @property
    def num_maxcc_nodes(self) -> int:
        return int(self.maxcc_local_idx.numel())


@dataclass(frozen=True)
class PairLabelArtifacts:
    pair1_compat_label: np.ndarray
    pair1_compat_mask: np.ndarray
    pair2_compat_label: np.ndarray
    pair2_compat_mask: np.ndarray


@dataclass(frozen=True)
class FeatureArtifacts:
    node_features: np.ndarray
    pair1_features: np.ndarray
    pair2_features: np.ndarray
    node_stat_map: Dict[str, np.ndarray]
