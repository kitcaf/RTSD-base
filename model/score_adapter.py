from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .data_types import ContextGraphSample


@dataclass(frozen=True)
class AdaptedScores:
    y_true: np.ndarray
    y_score: np.ndarray
    y_pred: np.ndarray
    node_indices: np.ndarray
    predicted_count: int
    true_count: int


class ScoreAdapter:
    def adapt(
        self,
        sample: ContextGraphSample,
        selected_nodes: np.ndarray,
        boosted_scores: np.ndarray,
    ) -> AdaptedScores:
        maxcc_positions = sample.maxcc_local_idx.detach().cpu().numpy()
        y_true = sample.y_root[maxcc_positions].detach().cpu().numpy()
        y_score = np.asarray(boosted_scores, dtype=np.float32)
        selected_set = set(int(node) for node in selected_nodes.tolist())
        y_pred = np.asarray(
            [1 if int(local_idx) in selected_set else 0 for local_idx in maxcc_positions.tolist()],
            dtype=np.int64,
        )
        return AdaptedScores(
            y_true=y_true,
            y_score=y_score,
            y_pred=y_pred,
            node_indices=sample.maxcc_orig_node_ids.detach().cpu().numpy(),
            predicted_count=int(y_pred.sum()),
            true_count=int(y_true.sum()),
        )
