from __future__ import annotations

from collections import deque
from typing import Dict

import networkx as nx
import numpy as np
from scipy.stats import rankdata

from .data_types import FeatureArtifacts, GraphStaticContext


class FeatureBuilder:
    NODE_FEATURE_NAMES = (
        "is_infected",
        "is_in_maxcc",
        "is_boundary_ring",
        "global_degree",
        "log_degree",
        "degree_percentile",
        "uninfected_neighbor_count",
        "boundary_exposure",
        "infected_degree_in_maxcc",
        "infected_neighbor_ratio_in_maxcc",
        "two_hop_infected_reach",
        "bridge_score",
        "outward_ratio",
        "stronger_neighbor_ratio",
        "kinf_vs_best_neighbor_ratio",
        "closeness_in_maxcc",
        "harmonic_in_maxcc",
        "eccentricity_score_in_maxcc",
        "distance_to_boundary",
        "local_peak_flag",
    )
    NODE_FEATURE_INDEX = {name: idx for idx, name in enumerate(NODE_FEATURE_NAMES)}

    PAIR_FEATURE_NAMES = (
        "degree_diff",
        "infected_degree_diff",
        "outward_ratio_diff",
        "closeness_diff",
        "shared_infected_neighbor_count",
        "same_two_hop_block",
    )

    def __init__(self, graph_context: GraphStaticContext) -> None:
        self.graph_context = graph_context

    @staticmethod
    def _log_normalize(values: np.ndarray | float, max_value: float) -> np.ndarray | float:
        denom = np.log1p(max(max_value, 1.0))
        return np.log1p(values) / denom

    @staticmethod
    def _safe_ratio(numerator: float, denominator: float) -> float:
        if denominator <= 0:
            return 0.0
        return float(numerator / denominator)

    @staticmethod
    def _descending_percentiles(values: np.ndarray) -> np.ndarray:
        if values.size == 0:
            return np.zeros(0, dtype=np.float32)
        if values.size == 1:
            return np.ones(1, dtype=np.float32)
        ranks = rankdata(-values, method="average")
        return (1.0 - (ranks - 1.0) / max(values.size - 1, 1)).astype(np.float32)

    def _compute_boundary_depth(self, maxcc_nodes: np.ndarray, infected_set: set[int]) -> Dict[int, float]:
        maxcc_set = set(int(node) for node in maxcc_nodes.tolist())
        boundary_nodes = [
            int(node)
            for node in maxcc_nodes
            if any(int(neighbor) not in infected_set for neighbor in self.graph_context.neighbors_list[int(node)])
        ]
        if not boundary_nodes:
            boundary_nodes = [int(node) for node in maxcc_nodes]

        distance_map = {int(node): np.inf for node in maxcc_nodes}
        queue = deque(boundary_nodes)
        for node in boundary_nodes:
            distance_map[node] = 0.0

        while queue:
            current = queue.popleft()
            current_distance = distance_map[current]
            for neighbor in self.graph_context.neighbors_list[current]:
                neighbor = int(neighbor)
                if neighbor not in maxcc_set:
                    continue
                if distance_map[neighbor] > current_distance + 1.0:
                    distance_map[neighbor] = current_distance + 1.0
                    queue.append(neighbor)

        finite_values = [value for value in distance_map.values() if np.isfinite(value)]
        max_depth = max(finite_values, default=0.0)
        denom = max(max_depth, 1.0)
        return {
            node: float(value / denom) if np.isfinite(value) else 0.0
            for node, value in distance_map.items()
        }

    def _compute_two_hop_reach(self, node: int, maxcc_set: set[int]) -> tuple[int, bool]:
        first_hop = {
            int(neighbor)
            for neighbor in self.graph_context.neighbors_list[node]
            if int(neighbor) in maxcc_set
        }
        two_hop = set(first_hop)
        has_two_hop_block = False
        for neighbor in first_hop:
            bridge_candidates = {
                int(next_hop)
                for next_hop in self.graph_context.neighbors_list[neighbor]
                if int(next_hop) in maxcc_set and int(next_hop) != node
            }
            if bridge_candidates:
                has_two_hop_block = True
            two_hop.update(bridge_candidates)
        two_hop.discard(node)
        return len(two_hop), has_two_hop_block

    def _shared_infected_neighbors(self, u: int, v: int, maxcc_set: set[int]) -> int:
        neighbors_u = {
            int(neighbor)
            for neighbor in self.graph_context.neighbors_list[u]
            if int(neighbor) in maxcc_set
        }
        neighbors_v = {
            int(neighbor)
            for neighbor in self.graph_context.neighbors_list[v]
            if int(neighbor) in maxcc_set
        }
        return len(neighbors_u & neighbors_v)

    def build(
        self,
        context_nodes: np.ndarray,
        maxcc_nodes: np.ndarray,
        pair1_index: np.ndarray,
        pair2_index: np.ndarray,
    ) -> FeatureArtifacts:
        context_nodes = np.asarray(context_nodes, dtype=np.int64)
        maxcc_nodes = np.asarray(maxcc_nodes, dtype=np.int64)
        context_index = {int(node): idx for idx, node in enumerate(context_nodes.tolist())}
        maxcc_set = set(int(node) for node in maxcc_nodes.tolist())
        infected_set = set(maxcc_set)

        maxcc_graph = self.graph_context.adjacency_csr[maxcc_nodes][:, maxcc_nodes]
        if hasattr(nx, "from_scipy_sparse_array"):
            nx_graph = nx.from_scipy_sparse_array(maxcc_graph)
        else:
            nx_graph = nx.from_scipy_sparse_matrix(maxcc_graph)

        closeness_lookup = nx.closeness_centrality(nx_graph) if len(maxcc_nodes) > 1 else {0: 1.0}
        harmonic_lookup = nx.harmonic_centrality(nx_graph) if len(maxcc_nodes) > 1 else {0: 1.0}
        try:
            eccentricity_lookup = nx.eccentricity(nx_graph) if len(maxcc_nodes) > 1 else {0: 0.0}
        except nx.NetworkXError:
            eccentricity_lookup = {node_idx: 0.0 for node_idx in range(len(maxcc_nodes))}

        max_harmonic = max(harmonic_lookup.values(), default=1.0) or 1.0
        max_eccentricity = max(eccentricity_lookup.values(), default=1.0) or 1.0

        maxcc_degree = np.zeros(len(context_nodes), dtype=np.float32)
        outward_ratio = np.zeros(len(context_nodes), dtype=np.float32)
        stronger_neighbor_ratio = np.zeros(len(context_nodes), dtype=np.float32)
        kinf_vs_best_neighbor_ratio = np.zeros(len(context_nodes), dtype=np.float32)
        closeness_in_maxcc = np.zeros(len(context_nodes), dtype=np.float32)
        harmonic_in_maxcc = np.zeros(len(context_nodes), dtype=np.float32)
        eccentricity_score = np.zeros(len(context_nodes), dtype=np.float32)
        distance_to_boundary = np.zeros(len(context_nodes), dtype=np.float32)
        local_peak = np.zeros(len(context_nodes), dtype=np.float32)
        two_hop_reach = np.zeros(len(context_nodes), dtype=np.float32)
        bridge_score = np.zeros(len(context_nodes), dtype=np.float32)
        maxcc_neighbor_ratio = np.zeros(len(context_nodes), dtype=np.float32)
        two_hop_block_flag = np.zeros(len(context_nodes), dtype=np.float32)

        boundary_depth_lookup = self._compute_boundary_depth(maxcc_nodes, infected_set)
        maxcc_node_order = {int(node): order for order, node in enumerate(maxcc_nodes.tolist())}

        for global_node in maxcc_nodes.tolist():
            local_idx = context_index[int(global_node)]
            node_degree = float(self.graph_context.degrees[int(global_node)])
            maxcc_neighbors = [
                int(neighbor)
                for neighbor in self.graph_context.neighbors_list[int(global_node)]
                if int(neighbor) in maxcc_set
            ]
            maxcc_deg_value = float(len(maxcc_neighbors))
            maxcc_degree[local_idx] = maxcc_deg_value
            maxcc_neighbor_ratio[local_idx] = self._safe_ratio(maxcc_deg_value, max(node_degree, 1.0))

            if maxcc_neighbors:
                smaller_degree_count = sum(
                    1 for neighbor in maxcc_neighbors
                    if self.graph_context.degrees[int(neighbor)] < self.graph_context.degrees[int(global_node)]
                )
                outward_ratio[local_idx] = self._safe_ratio(smaller_degree_count, len(maxcc_neighbors))

                stronger_count = sum(
                    1
                    for neighbor in maxcc_neighbors
                    if len([
                        1
                        for next_neighbor in self.graph_context.neighbors_list[int(neighbor)]
                        if int(next_neighbor) in maxcc_set
                    ]) > maxcc_deg_value
                )
                stronger_neighbor_ratio[local_idx] = self._safe_ratio(stronger_count, len(maxcc_neighbors))

                best_neighbor_kinf = max(
                    len([
                        1
                        for next_neighbor in self.graph_context.neighbors_list[int(neighbor)]
                        if int(next_neighbor) in maxcc_set
                    ])
                    for neighbor in maxcc_neighbors
                )
                kinf_vs_best_neighbor_ratio[local_idx] = self._safe_ratio(maxcc_deg_value, max(best_neighbor_kinf, 1))
                local_peak[local_idx] = 1.0 if maxcc_deg_value >= best_neighbor_kinf else 0.0

                sum_neighbor_kinf = sum(
                    len([
                        1
                        for next_neighbor in self.graph_context.neighbors_list[int(neighbor)]
                        if int(next_neighbor) in maxcc_set
                    ])
                    for neighbor in maxcc_neighbors
                )
                bridge_score[local_idx] = self._safe_ratio(maxcc_deg_value * max(node_degree, 1.0), sum_neighbor_kinf + 1e-6)
            else:
                kinf_vs_best_neighbor_ratio[local_idx] = 1.0
                local_peak[local_idx] = 1.0

            two_hop_count, has_two_hop_block = self._compute_two_hop_reach(int(global_node), maxcc_set)
            two_hop_reach[local_idx] = self._safe_ratio(two_hop_count, max(len(maxcc_nodes) - 1, 1))
            two_hop_block_flag[local_idx] = 1.0 if has_two_hop_block else 0.0

            node_order = maxcc_node_order[int(global_node)]
            closeness_in_maxcc[local_idx] = float(closeness_lookup.get(node_order, 0.0))
            harmonic_in_maxcc[local_idx] = self._safe_ratio(float(harmonic_lookup.get(node_order, 0.0)), max_harmonic)
            eccentricity_in_graph = float(eccentricity_lookup.get(node_order, max_eccentricity))
            eccentricity_score[local_idx] = 1.0 - self._safe_ratio(eccentricity_in_graph, max_eccentricity)
            distance_to_boundary[local_idx] = float(boundary_depth_lookup.get(int(global_node), 0.0))

        maxcc_degree_scale = max(float(maxcc_degree.max()) if maxcc_degree.size > 0 else 0.0, 1.0)
        bridge_scale = max(float(bridge_score.max()) if bridge_score.size > 0 else 0.0, 1.0)
        node_features = np.zeros((len(context_nodes), len(self.NODE_FEATURE_NAMES)), dtype=np.float32)

        for local_idx, global_node in enumerate(context_nodes.tolist()):
            degree_value = float(self.graph_context.degrees[int(global_node)])
            uninfected_count = sum(
                1
                for neighbor in self.graph_context.neighbors_list[int(global_node)]
                if int(neighbor) not in infected_set
            )

            node_features[local_idx, self.NODE_FEATURE_INDEX["is_infected"]] = 1.0 if int(global_node) in maxcc_set else 0.0
            node_features[local_idx, self.NODE_FEATURE_INDEX["is_in_maxcc"]] = 1.0 if int(global_node) in maxcc_set else 0.0
            node_features[local_idx, self.NODE_FEATURE_INDEX["is_boundary_ring"]] = 0.0 if int(global_node) in maxcc_set else 1.0
            node_features[local_idx, self.NODE_FEATURE_INDEX["global_degree"]] = self._safe_ratio(degree_value, self.graph_context.max_degree)
            node_features[local_idx, self.NODE_FEATURE_INDEX["log_degree"]] = float(self._log_normalize(degree_value, self.graph_context.max_degree))
            node_features[local_idx, self.NODE_FEATURE_INDEX["degree_percentile"]] = float(self.graph_context.degree_percentile[int(global_node)])
            node_features[local_idx, self.NODE_FEATURE_INDEX["uninfected_neighbor_count"]] = float(self._log_normalize(uninfected_count, self.graph_context.max_degree))
            node_features[local_idx, self.NODE_FEATURE_INDEX["boundary_exposure"]] = self._safe_ratio(uninfected_count, max(degree_value, 1.0))
            node_features[local_idx, self.NODE_FEATURE_INDEX["infected_degree_in_maxcc"]] = self._safe_ratio(maxcc_degree[local_idx], maxcc_degree_scale)
            node_features[local_idx, self.NODE_FEATURE_INDEX["infected_neighbor_ratio_in_maxcc"]] = maxcc_neighbor_ratio[local_idx]
            node_features[local_idx, self.NODE_FEATURE_INDEX["two_hop_infected_reach"]] = two_hop_reach[local_idx]
            node_features[local_idx, self.NODE_FEATURE_INDEX["bridge_score"]] = self._safe_ratio(bridge_score[local_idx], bridge_scale)
            node_features[local_idx, self.NODE_FEATURE_INDEX["outward_ratio"]] = outward_ratio[local_idx]
            node_features[local_idx, self.NODE_FEATURE_INDEX["stronger_neighbor_ratio"]] = stronger_neighbor_ratio[local_idx]
            node_features[local_idx, self.NODE_FEATURE_INDEX["kinf_vs_best_neighbor_ratio"]] = kinf_vs_best_neighbor_ratio[local_idx]
            node_features[local_idx, self.NODE_FEATURE_INDEX["closeness_in_maxcc"]] = closeness_in_maxcc[local_idx]
            node_features[local_idx, self.NODE_FEATURE_INDEX["harmonic_in_maxcc"]] = harmonic_in_maxcc[local_idx]
            node_features[local_idx, self.NODE_FEATURE_INDEX["eccentricity_score_in_maxcc"]] = eccentricity_score[local_idx]
            node_features[local_idx, self.NODE_FEATURE_INDEX["distance_to_boundary"]] = distance_to_boundary[local_idx]
            node_features[local_idx, self.NODE_FEATURE_INDEX["local_peak_flag"]] = local_peak[local_idx]

        pair1_features = self._build_pair_features(
            pair_index=pair1_index,
            context_nodes=context_nodes,
            node_features=node_features,
            maxcc_set=maxcc_set,
            two_hop_block_flag=two_hop_block_flag,
        )
        pair2_features = self._build_pair_features(
            pair_index=pair2_index,
            context_nodes=context_nodes,
            node_features=node_features,
            maxcc_set=maxcc_set,
            two_hop_block_flag=two_hop_block_flag,
        )
        node_stat_map = {
            "infected_degree_in_maxcc": node_features[:, self.NODE_FEATURE_INDEX["infected_degree_in_maxcc"]],
            "closeness_in_maxcc": node_features[:, self.NODE_FEATURE_INDEX["closeness_in_maxcc"]],
            "outward_ratio": node_features[:, self.NODE_FEATURE_INDEX["outward_ratio"]],
        }
        return FeatureArtifacts(
            node_features=node_features,
            pair1_features=pair1_features,
            pair2_features=pair2_features,
            node_stat_map=node_stat_map,
        )

    def _build_pair_features(
        self,
        pair_index: np.ndarray,
        context_nodes: np.ndarray,
        node_features: np.ndarray,
        maxcc_set: set[int],
        two_hop_block_flag: np.ndarray,
    ) -> np.ndarray:
        if pair_index.size == 0:
            return np.zeros((0, len(self.PAIR_FEATURE_NAMES)), dtype=np.float32)

        pair_features = np.zeros((pair_index.shape[1], len(self.PAIR_FEATURE_NAMES)), dtype=np.float32)
        degree_idx = self.NODE_FEATURE_INDEX["global_degree"]
        infected_degree_idx = self.NODE_FEATURE_INDEX["infected_degree_in_maxcc"]
        outward_idx = self.NODE_FEATURE_INDEX["outward_ratio"]
        closeness_idx = self.NODE_FEATURE_INDEX["closeness_in_maxcc"]

        for pair_pos in range(pair_index.shape[1]):
            local_u = int(pair_index[0, pair_pos])
            local_v = int(pair_index[1, pair_pos])
            global_u = int(context_nodes[local_u])
            global_v = int(context_nodes[local_v])

            pair_features[pair_pos, 0] = node_features[local_u, degree_idx] - node_features[local_v, degree_idx]
            pair_features[pair_pos, 1] = node_features[local_u, infected_degree_idx] - node_features[local_v, infected_degree_idx]
            pair_features[pair_pos, 2] = node_features[local_u, outward_idx] - node_features[local_v, outward_idx]
            pair_features[pair_pos, 3] = node_features[local_u, closeness_idx] - node_features[local_v, closeness_idx]
            pair_features[pair_pos, 4] = float(
                self._shared_infected_neighbors(global_u, global_v, maxcc_set) / max(len(maxcc_set), 1)
            )
            pair_features[pair_pos, 5] = 1.0 if (two_hop_block_flag[local_u] > 0.0 and two_hop_block_flag[local_v] > 0.0) else 0.0

        return pair_features.astype(np.float32)
