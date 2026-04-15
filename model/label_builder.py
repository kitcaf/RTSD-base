from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np

from .data_types import PairLabelArtifacts


@dataclass(frozen=True)
class LabelBuilderConfig:
    pair1_negative_ratio: int = 3
    pair2_negative_ratio: int = 3


class LabelBuilder:
    HARD_NEGATIVE_KEYS = (
        "infected_degree_in_maxcc",
        "closeness_in_maxcc",
        "outward_ratio",
    )

    def __init__(self, config: LabelBuilderConfig | None = None, seed: int = 3407) -> None:
        self.config = config or LabelBuilderConfig()
        self.rng = np.random.default_rng(seed)

    def build_root_labels(
        self,
        context_nodes: np.ndarray,
        maxcc_set: set[int],
        source_set: set[int],
    ) -> np.ndarray:
        labels = np.zeros(len(context_nodes), dtype=np.float32)
        for idx, node in enumerate(context_nodes.tolist()):
            labels[idx] = 1.0 if (int(node) in maxcc_set and int(node) in source_set) else 0.0
        return labels

    def build_pair_labels(
        self,
        pair1_index: np.ndarray,
        pair2_index: np.ndarray,
        context_nodes: np.ndarray,
        source_set: set[int],
        node_stat_map: Dict[str, np.ndarray] | None = None,
    ) -> PairLabelArtifacts:
        source_membership = np.asarray(
            [1 if int(node) in source_set else 0 for node in context_nodes.tolist()],
            dtype=np.int64,
        )
        hard_scores = self._build_hard_negative_scores(
            node_stat_map=node_stat_map,
            num_context_nodes=len(context_nodes),
        )

        pair1_compat_label, pair1_compat_mask = self._build_compatibility_targets(
            pair_index=pair1_index,
            source_membership=source_membership,
            hard_scores=hard_scores,
            negative_ratio=self.config.pair1_negative_ratio,
        )
        pair2_compat_label, pair2_compat_mask = self._build_compatibility_targets(
            pair_index=pair2_index,
            source_membership=source_membership,
            hard_scores=hard_scores,
            negative_ratio=self.config.pair2_negative_ratio,
        )

        return PairLabelArtifacts(
            pair1_compat_label=pair1_compat_label,
            pair1_compat_mask=pair1_compat_mask,
            pair2_compat_label=pair2_compat_label,
            pair2_compat_mask=pair2_compat_mask,
        )

    def _build_hard_negative_scores(
        self,
        node_stat_map: Dict[str, np.ndarray] | None,
        num_context_nodes: int,
    ) -> np.ndarray:
        if not node_stat_map:
            return np.zeros(num_context_nodes, dtype=np.float32)

        score_components = []
        for key in self.HARD_NEGATIVE_KEYS:
            values = node_stat_map.get(key)
            if values is None:
                continue
            values = np.asarray(values, dtype=np.float32).reshape(-1)
            if values.size != num_context_nodes:
                continue
            score_components.append(values)

        if not score_components:
            return np.zeros(num_context_nodes, dtype=np.float32)
        return np.mean(np.stack(score_components, axis=0), axis=0).astype(np.float32)

    def _build_compatibility_targets(
        self,
        pair_index: np.ndarray,
        source_membership: np.ndarray,
        hard_scores: np.ndarray,
        negative_ratio: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        num_pairs = int(pair_index.shape[1]) if pair_index.ndim == 2 else 0
        compat_label = np.zeros(num_pairs, dtype=np.float32)
        compat_mask = np.zeros(num_pairs, dtype=bool)
        if num_pairs == 0:
            return compat_label, compat_mask

        positive_positions: list[int] = []
        mixed_negative_positions: list[int] = []
        mixed_negative_scores: list[float] = []
        hard_negative_positions: list[int] = []
        hard_negative_scores: list[float] = []

        for pair_pos in range(num_pairs):
            local_u = int(pair_index[0, pair_pos])
            local_v = int(pair_index[1, pair_pos])
            is_u_source = bool(source_membership[local_u])
            is_v_source = bool(source_membership[local_v])
            if is_u_source and is_v_source:
                compat_label[pair_pos] = 1.0
                positive_positions.append(pair_pos)
                continue

            pair_score = float((hard_scores[local_u] + hard_scores[local_v]) * 0.5)
            if is_u_source ^ is_v_source:
                mixed_negative_positions.append(pair_pos)
                mixed_negative_scores.append(pair_score)
            else:
                hard_negative_positions.append(pair_pos)
                hard_negative_scores.append(pair_score)

        if positive_positions:
            compat_mask[np.asarray(positive_positions, dtype=np.int64)] = True

        negative_budget = max(negative_ratio * max(len(positive_positions), 1), negative_ratio)
        selected_negative_positions = self._select_negative_positions(
            mixed_positions=mixed_negative_positions,
            mixed_scores=mixed_negative_scores,
            hard_positions=hard_negative_positions,
            hard_scores=hard_negative_scores,
            negative_budget=negative_budget,
        )
        if selected_negative_positions:
            compat_mask[np.asarray(selected_negative_positions, dtype=np.int64)] = True
        return compat_label, compat_mask

    def _select_negative_positions(
        self,
        mixed_positions: list[int],
        mixed_scores: list[float],
        hard_positions: list[int],
        hard_scores: list[float],
        negative_budget: int,
    ) -> list[int]:
        if negative_budget <= 0:
            return []

        selected_positions: list[int] = []
        remaining_budget = int(negative_budget)

        ranked_mixed_positions = self._rank_positions_by_score(mixed_positions, mixed_scores)
        if ranked_mixed_positions:
            take_count = min(remaining_budget, len(ranked_mixed_positions))
            selected_positions.extend(ranked_mixed_positions[:take_count])
            remaining_budget -= take_count

        if remaining_budget > 0:
            ranked_hard_positions = self._rank_positions_by_score(hard_positions, hard_scores)
            take_count = min(remaining_budget, len(ranked_hard_positions))
            selected_positions.extend(ranked_hard_positions[:take_count])

        return selected_positions

    def _rank_positions_by_score(self, positions: list[int], scores: list[float]) -> list[int]:
        if not positions:
            return []
        order = np.argsort(-np.asarray(scores, dtype=np.float32), kind="stable")
        return [int(positions[int(idx)]) for idx in order.tolist()]
