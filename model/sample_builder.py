from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import scipy.sparse as sp
import torch
from scipy.sparse.csgraph import connected_components, shortest_path
from scipy.stats import rankdata

from .data_types import ContextGraphSample, GraphStaticContext
from .feature_builder import FeatureBuilder
from .label_builder import LabelBuilder


@dataclass(frozen=True)
class SampleBuilderStats:
    total_cascades: int
    kept_samples: int
    skipped_empty: int
    skipped_without_source_in_maxcc: int


class ContextGraphDatasetBuilder:
    def __init__(
        self,
        adjacency_matrix,
        include_ring_hops: int = 1,
        max_boundary_ring_size: int | None = None,
        allow_zero_source_in_maxcc: bool = False,
        seed: int = 3407,
    ) -> None:
        adjacency_csr = adjacency_matrix if sp.issparse(adjacency_matrix) else sp.csr_matrix(adjacency_matrix)
        adjacency_csr = adjacency_csr.tocsr()
        degrees = np.asarray(adjacency_csr.getnnz(axis=1)).astype(np.float32)
        degree_percentile = (rankdata(degrees, method="average") - 1.0) / max(len(degrees) - 1, 1)

        self.graph_context = GraphStaticContext(
            adjacency_csr=adjacency_csr,
            neighbors_list=tuple(
                adjacency_csr.indices[adjacency_csr.indptr[i]:adjacency_csr.indptr[i + 1]].copy()
                for i in range(adjacency_csr.shape[0])
            ),
            degrees=degrees,
            degree_percentile=degree_percentile.astype(np.float32),
            max_degree=max(float(degrees.max()) if degrees.size > 0 else 0.0, 1.0),
            num_nodes=int(adjacency_csr.shape[0]),
        )
        self.include_ring_hops = include_ring_hops
        self.max_boundary_ring_size = max_boundary_ring_size
        self.allow_zero_source_in_maxcc = allow_zero_source_in_maxcc
        self.feature_builder = FeatureBuilder(self.graph_context)
        self.label_builder = LabelBuilder(seed=seed)

    def _extract_maxcc_nodes(self, infected_nodes: np.ndarray) -> np.ndarray:
        if infected_nodes.size == 0:
            return np.zeros(0, dtype=np.int64)
        if infected_nodes.size == 1:
            return infected_nodes.astype(np.int64)

        infected_subgraph = self.graph_context.adjacency_csr[infected_nodes][:, infected_nodes]
        component_count, component_labels = connected_components(infected_subgraph, directed=False)
        if component_count <= 0:
            return np.zeros(0, dtype=np.int64)
        component_sizes = np.bincount(component_labels)
        largest_component = int(np.argmax(component_sizes))
        return infected_nodes[component_labels == largest_component].astype(np.int64)

    def _build_ring_neighbors(self, maxcc_nodes: np.ndarray) -> np.ndarray:
        current_shell = set(int(node) for node in maxcc_nodes.tolist())
        visited = set(current_shell)
        ring_nodes = set()
        for _ in range(max(self.include_ring_hops, 0)):
            next_shell = set()
            for node in current_shell:
                for neighbor in self.graph_context.neighbors_list[node]:
                    neighbor = int(neighbor)
                    if neighbor in visited:
                        continue
                    visited.add(neighbor)
                    next_shell.add(neighbor)
            ring_nodes.update(next_shell)
            current_shell = next_shell

        ring_array = np.asarray(sorted(ring_nodes), dtype=np.int64)
        if self.max_boundary_ring_size is None or ring_array.size <= self.max_boundary_ring_size:
            return ring_array

        maxcc_set = set(int(node) for node in maxcc_nodes.tolist())
        scored_ring = []
        for ring_node in ring_array.tolist():
            overlap = sum(
                1
                for neighbor in self.graph_context.neighbors_list[int(ring_node)]
                if int(neighbor) in maxcc_set
            )
            scored_ring.append((overlap, -self.graph_context.degrees[int(ring_node)], int(ring_node)))
        scored_ring.sort(reverse=True)
        selected = [node for _, _, node in scored_ring[: self.max_boundary_ring_size]]
        return np.asarray(sorted(selected), dtype=np.int64)

    def _build_local_edge_index(self, context_nodes: np.ndarray) -> np.ndarray:
        if context_nodes.size == 0:
            return np.zeros((2, 0), dtype=np.int64)
        subgraph = self.graph_context.adjacency_csr[context_nodes][:, context_nodes].tocoo()
        if subgraph.nnz == 0:
            return np.zeros((2, 0), dtype=np.int64)
        return np.vstack([subgraph.row.astype(np.int64), subgraph.col.astype(np.int64)])

    def _build_pair_indices(self, maxcc_nodes: np.ndarray, context_index: dict[int, int]) -> tuple[np.ndarray, np.ndarray]:
        if maxcc_nodes.size <= 1:
            empty = np.zeros((2, 0), dtype=np.int64)
            return empty, empty

        maxcc_subgraph = self.graph_context.adjacency_csr[maxcc_nodes][:, maxcc_nodes]
        maxcc_coo = maxcc_subgraph.tocoo()
        pair1_edges = {
            tuple(sorted((int(row), int(col))))
            for row, col in zip(maxcc_coo.row.tolist(), maxcc_coo.col.tolist())
            if int(row) < int(col)
        }
        pair1_index = np.zeros((2, len(pair1_edges)), dtype=np.int64)
        for pair_pos, (local_u, local_v) in enumerate(sorted(pair1_edges)):
            global_u = int(maxcc_nodes[local_u])
            global_v = int(maxcc_nodes[local_v])
            pair1_index[:, pair_pos] = np.array([context_index[global_u], context_index[global_v]], dtype=np.int64)

        distance_matrix = shortest_path(maxcc_subgraph, directed=False, unweighted=True, return_predecessors=False)
        pair2_edges = []
        for local_u in range(len(maxcc_nodes)):
            for local_v in range(local_u + 1, len(maxcc_nodes)):
                if int(distance_matrix[local_u, local_v]) == 2:
                    global_u = int(maxcc_nodes[local_u])
                    global_v = int(maxcc_nodes[local_v])
                    pair2_edges.append((context_index[global_u], context_index[global_v]))

        pair2_index = np.zeros((2, len(pair2_edges)), dtype=np.int64)
        for pair_pos, (local_u, local_v) in enumerate(pair2_edges):
            pair2_index[:, pair_pos] = np.array([local_u, local_v], dtype=np.int64)
        return pair1_index, pair2_index

    def build_dataset(self, influence_matrices) -> tuple[List[ContextGraphSample], SampleBuilderStats]:
        samples: List[ContextGraphSample] = []
        skipped_empty = 0
        skipped_without_source_in_maxcc = 0

        for cascade_id, cascade_matrix in enumerate(influence_matrices):
            sample = self.build_single_sample(cascade_id, cascade_matrix)
            if sample is None:
                source_nodes = np.flatnonzero(np.asarray(cascade_matrix)[:, 0]).astype(np.int64)
                infected_nodes = np.flatnonzero(np.asarray(cascade_matrix)[:, 1]).astype(np.int64)
                if source_nodes.size == 0 or infected_nodes.size == 0:
                    skipped_empty += 1
                else:
                    skipped_without_source_in_maxcc += 1
                continue
            samples.append(sample)

        return samples, SampleBuilderStats(
            total_cascades=len(influence_matrices),
            kept_samples=len(samples),
            skipped_empty=skipped_empty,
            skipped_without_source_in_maxcc=skipped_without_source_in_maxcc,
        )

    def build_single_sample(self, cascade_id: int, cascade_matrix) -> Optional[ContextGraphSample]:
        cascade_matrix = np.asarray(cascade_matrix)
        if cascade_matrix.ndim != 2 or cascade_matrix.shape[1] < 2:
            raise ValueError("Expected cascade matrix with shape [N, 2] or wider.")

        source_nodes = np.flatnonzero(cascade_matrix[:, 0]).astype(np.int64)
        infected_nodes = np.flatnonzero(cascade_matrix[:, 1]).astype(np.int64)
        if source_nodes.size == 0 or infected_nodes.size == 0:
            return None

        maxcc_nodes = self._extract_maxcc_nodes(infected_nodes)
        if maxcc_nodes.size == 0:
            return None

        source_in_maxcc = np.intersect1d(source_nodes, maxcc_nodes, assume_unique=False)
        if source_in_maxcc.size == 0 and not self.allow_zero_source_in_maxcc:
            return None

        ring_nodes = self._build_ring_neighbors(maxcc_nodes)
        context_nodes = np.asarray(sorted(set(maxcc_nodes.tolist()) | set(ring_nodes.tolist())), dtype=np.int64)
        context_index = {int(node): idx for idx, node in enumerate(context_nodes.tolist())}
        maxcc_local_idx = np.asarray([context_index[int(node)] for node in maxcc_nodes.tolist()], dtype=np.int64)

        edge_index_ctx = self._build_local_edge_index(context_nodes)
        pair1_index, pair2_index = self._build_pair_indices(maxcc_nodes, context_index)

        feature_artifacts = self.feature_builder.build(
            context_nodes=context_nodes,
            maxcc_nodes=maxcc_nodes,
            pair1_index=pair1_index,
            pair2_index=pair2_index,
        )
        label_artifacts = self.label_builder.build_pair_labels(
            pair1_index=pair1_index,
            pair2_index=pair2_index,
            context_nodes=context_nodes,
            source_set=set(int(node) for node in source_nodes.tolist()),
            node_stat_map=feature_artifacts.node_stat_map,
        )
        y_root = self.label_builder.build_root_labels(
            context_nodes=context_nodes,
            maxcc_set=set(int(node) for node in maxcc_nodes.tolist()),
            source_set=set(int(node) for node in source_nodes.tolist()),
        )

        maxcc_mask = np.zeros(len(context_nodes), dtype=bool)
        maxcc_mask[maxcc_local_idx] = True

        return ContextGraphSample(
            cascade_id=int(cascade_id),
            orig_node_ids=torch.as_tensor(context_nodes, dtype=torch.long),
            edge_index_ctx=torch.as_tensor(edge_index_ctx, dtype=torch.long),
            x=torch.as_tensor(feature_artifacts.node_features, dtype=torch.float32),
            maxcc_mask=torch.as_tensor(maxcc_mask, dtype=torch.bool),
            loss_mask=torch.as_tensor(maxcc_mask, dtype=torch.bool),
            y_root=torch.as_tensor(y_root, dtype=torch.float32),
            k_label=int(source_in_maxcc.size),
            pair1_index=torch.as_tensor(pair1_index, dtype=torch.long),
            pair1_features=torch.as_tensor(feature_artifacts.pair1_features, dtype=torch.float32),
            pair1_compat_label=torch.as_tensor(label_artifacts.pair1_compat_label, dtype=torch.float32),
            pair1_compat_mask=torch.as_tensor(label_artifacts.pair1_compat_mask, dtype=torch.bool),
            pair2_index=torch.as_tensor(pair2_index, dtype=torch.long),
            pair2_features=torch.as_tensor(feature_artifacts.pair2_features, dtype=torch.float32),
            pair2_compat_label=torch.as_tensor(label_artifacts.pair2_compat_label, dtype=torch.float32),
            pair2_compat_mask=torch.as_tensor(label_artifacts.pair2_compat_mask, dtype=torch.bool),
            maxcc_local_idx=torch.as_tensor(maxcc_local_idx, dtype=torch.long),
            maxcc_orig_node_ids=torch.as_tensor(maxcc_nodes, dtype=torch.long),
        )
