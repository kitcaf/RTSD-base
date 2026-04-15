from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import DecoderConfig
from .data_types import ContextGraphSample


@dataclass(frozen=True)
class DecoderLosses:
    step_loss: torch.Tensor
    set_loss: torch.Tensor


@dataclass(frozen=True)
class DecodeResult:
    selected_nodes: np.ndarray
    node_scores: np.ndarray
    selected_count: int


def _inverse_softplus(value: float) -> float:
    clipped = max(float(value), 1e-6)
    return float(np.log(np.expm1(clipped)))


class SetDecoder(nn.Module):
    """Learned set decoder with a submodular backbone and local attractive residual."""

    def __init__(self, config: DecoderConfig, hidden_dim: int, node_feature_dim: int) -> None:
        super().__init__()
        self.config = config
        decoder_hidden_dim = int(config.decoder_hidden_dim)

        self.source_proj = nn.Sequential(
            nn.Linear(hidden_dim + node_feature_dim + 1, decoder_hidden_dim),
            nn.ELU(),
            nn.Linear(decoder_hidden_dim, decoder_hidden_dim),
            nn.ELU(),
        )
        self.demand_proj = nn.Sequential(
            nn.Linear(hidden_dim + node_feature_dim, decoder_hidden_dim),
            nn.ELU(),
            nn.Linear(decoder_hidden_dim, decoder_hidden_dim),
            nn.ELU(),
        )
        coverage_input_dim = decoder_hidden_dim * 4 + 4
        self.coverage_head = nn.Sequential(
            nn.Linear(coverage_input_dim, decoder_hidden_dim),
            nn.ELU(),
            nn.Linear(decoder_hidden_dim, 1),
        )

        self.unary_scale_raw = nn.Parameter(torch.tensor(_inverse_softplus(config.unary_weight), dtype=torch.float32))
        self.coverage_scale_raw = nn.Parameter(
            torch.tensor(_inverse_softplus(config.coverage_weight), dtype=torch.float32)
        )
        self.pair1_scale_raw = nn.Parameter(
            torch.tensor(_inverse_softplus(config.compatibility_weight), dtype=torch.float32)
        )
        self.pair2_scale_raw = nn.Parameter(
            torch.tensor(_inverse_softplus(config.compatibility_weight * 0.6), dtype=torch.float32)
        )
        self.count_scale_raw = nn.Parameter(
            torch.tensor(_inverse_softplus(config.count_prior_weight), dtype=torch.float32)
        )

    def compute_losses(
        self,
        sample: ContextGraphSample,
        hidden_states: torch.Tensor,
        node_logits: torch.Tensor,
        compat1_logits: torch.Tensor,
        compat2_logits: torch.Tensor,
        count_logits: torch.Tensor,
    ) -> DecoderLosses:
        context = self._prepare_context(
            sample=sample,
            hidden_states=hidden_states,
            node_logits=node_logits,
            compat1_logits=compat1_logits,
            compat2_logits=compat2_logits,
            count_logits=count_logits,
            include_gold=True,
        )
        if context is None or int(context["gold_mask"].sum().item()) == 0:
            zero = hidden_states.new_zeros(())
            return DecoderLosses(step_loss=zero, set_loss=zero)

        return DecoderLosses(
            step_loss=self._step_loss(context),
            set_loss=self._set_loss(context),
        )

    def decode(
        self,
        sample: ContextGraphSample,
        hidden_states: torch.Tensor,
        node_logits: torch.Tensor,
        compat1_logits: torch.Tensor,
        compat2_logits: torch.Tensor,
        count_logits: torch.Tensor,
    ) -> DecodeResult:
        context = self._prepare_context(
            sample=sample,
            hidden_states=hidden_states,
            node_logits=node_logits,
            compat1_logits=compat1_logits,
            compat2_logits=compat2_logits,
            count_logits=count_logits,
            include_gold=False,
        )
        if context is None:
            base_scores = torch.sigmoid(node_logits[sample.maxcc_local_idx]).detach().cpu().numpy().astype(np.float32)
            return DecodeResult(
                selected_nodes=np.zeros(0, dtype=np.int64),
                node_scores=base_scores,
                selected_count=0,
            )

        best_mask, _ = self._best_prefix(context)
        selected_nodes = context["candidate_local_idx"][best_mask].detach().cpu().numpy().astype(np.int64)
        base_scores = torch.sigmoid(node_logits[sample.maxcc_local_idx]).detach().clone()
        candidate_positions = context["candidate_maxcc_positions"]
        base_scores[candidate_positions] = torch.maximum(base_scores[candidate_positions], context["singleton_scores"])
        return DecodeResult(
            selected_nodes=selected_nodes,
            node_scores=base_scores.detach().cpu().numpy().astype(np.float32),
            selected_count=int(best_mask.sum().item()),
        )

    def _prepare_context(
        self,
        sample: ContextGraphSample,
        hidden_states: torch.Tensor,
        node_logits: torch.Tensor,
        compat1_logits: torch.Tensor,
        compat2_logits: torch.Tensor,
        count_logits: torch.Tensor,
        include_gold: bool,
    ) -> dict | None:
        maxcc_local_idx = sample.maxcc_local_idx
        if maxcc_local_idx.numel() == 0:
            return None

        candidate_local_idx = self._build_candidate_pool(sample, node_logits, count_logits, include_gold=include_gold)
        if candidate_local_idx.numel() == 0:
            return None

        maxcc_position_lookup = {int(local_idx): pos for pos, local_idx in enumerate(maxcc_local_idx.tolist())}
        candidate_maxcc_positions = torch.as_tensor(
            [maxcc_position_lookup[int(local_idx)] for local_idx in candidate_local_idx.tolist()],
            dtype=torch.long,
            device=node_logits.device,
        )
        count_log_probs = self._count_log_probs(count_logits)
        coverage_matrix = self._build_coverage_matrix(
            sample=sample,
            hidden_states=hidden_states,
            node_logits=node_logits,
            candidate_local_idx=candidate_local_idx,
            maxcc_local_idx=maxcc_local_idx,
        )
        pair1_matrix, pair2_matrix = self._build_pair_matrices(
            sample=sample,
            candidate_local_idx=candidate_local_idx,
            compat1_logits=compat1_logits,
            compat2_logits=compat2_logits,
        )
        candidate_unary = torch.sigmoid(node_logits[candidate_local_idx])
        gold_mask = sample.y_root[candidate_local_idx] > 0.5
        singleton_scores = self._singleton_scores(
            candidate_unary=candidate_unary,
            coverage_matrix=coverage_matrix,
            count_log_probs=count_log_probs,
        )
        return {
            "candidate_local_idx": candidate_local_idx,
            "candidate_unary": candidate_unary,
            "candidate_maxcc_positions": candidate_maxcc_positions,
            "coverage_matrix": coverage_matrix,
            "pair1_matrix": pair1_matrix,
            "pair2_matrix": pair2_matrix,
            "gold_mask": gold_mask,
            "count_log_probs": count_log_probs,
            "singleton_scores": singleton_scores,
        }

    def _build_candidate_pool(
        self,
        sample: ContextGraphSample,
        node_logits: torch.Tensor,
        count_logits: torch.Tensor,
        include_gold: bool,
    ) -> torch.Tensor:
        maxcc_local_idx = sample.maxcc_local_idx
        if maxcc_local_idx.numel() == 0:
            return maxcc_local_idx.new_zeros((0,))

        count_mode = int(torch.argmax(self._count_log_probs(count_logits)).item())
        candidate_pool_size = min(
            self.config.candidate_pool_cap,
            self.config.candidate_pool_multiplier * max(count_mode, 1) + self.config.candidate_pool_bias,
            int(maxcc_local_idx.numel()),
        )
        candidate_pool_size = max(candidate_pool_size, 1)
        top_indices = torch.topk(node_logits[maxcc_local_idx], k=candidate_pool_size).indices
        top_candidates = maxcc_local_idx[top_indices]
        if not include_gold:
            return top_candidates

        gold_nodes = maxcc_local_idx[sample.y_root[maxcc_local_idx] > 0.5]
        merged: List[int] = []
        seen = set()
        for tensor in (top_candidates, gold_nodes):
            for node in tensor.tolist():
                node_int = int(node)
                if node_int in seen:
                    continue
                seen.add(node_int)
                merged.append(node_int)
        return torch.as_tensor(merged, dtype=torch.long, device=node_logits.device)

    def _build_coverage_matrix(
        self,
        sample: ContextGraphSample,
        hidden_states: torch.Tensor,
        node_logits: torch.Tensor,
        candidate_local_idx: torch.Tensor,
        maxcc_local_idx: torch.Tensor,
    ) -> torch.Tensor:
        candidate_features = sample.x[candidate_local_idx]
        demand_features = sample.x[maxcc_local_idx]
        candidate_states = hidden_states[candidate_local_idx]
        demand_states = hidden_states[maxcc_local_idx]
        candidate_probs = torch.sigmoid(node_logits[candidate_local_idx]).unsqueeze(-1)

        source_repr = self.source_proj(torch.cat([candidate_states, candidate_features, candidate_probs], dim=-1))
        demand_repr = self.demand_proj(torch.cat([demand_states, demand_features], dim=-1))
        distance_buckets = self._build_distance_buckets(
            candidate_local_idx=candidate_local_idx,
            maxcc_local_idx=maxcc_local_idx,
            pair1_index=sample.pair1_index,
            device=hidden_states.device,
        )
        relation_features = F.one_hot(distance_buckets, num_classes=4).to(dtype=hidden_states.dtype)

        source_expand = source_repr[:, None, :]
        demand_expand = demand_repr[None, :, :]
        source_tiled = source_expand.expand(-1, demand_repr.size(0), -1)
        demand_tiled = demand_expand.expand(source_repr.size(0), -1, -1)
        coverage_input = torch.cat(
            [
                source_tiled,
                demand_tiled,
                torch.abs(source_tiled - demand_tiled),
                source_tiled * demand_tiled,
                relation_features,
            ],
            dim=-1,
        )
        return F.softplus(self.coverage_head(coverage_input).squeeze(-1))

    def _build_distance_buckets(
        self,
        candidate_local_idx: torch.Tensor,
        maxcc_local_idx: torch.Tensor,
        pair1_index: torch.Tensor,
        device: torch.device,
    ) -> torch.Tensor:
        adjacency_lookup = self._build_undirected_adjacency(pair1_index, maxcc_local_idx)
        maxcc_nodes = [int(node) for node in maxcc_local_idx.tolist()]
        distance_buckets = np.full((candidate_local_idx.numel(), maxcc_local_idx.numel()), 3, dtype=np.int64)
        for row_idx, source_node in enumerate(candidate_local_idx.tolist()):
            distances = self._shortest_distances(source=int(source_node), adjacency_lookup=adjacency_lookup)
            for col_idx, target_node in enumerate(maxcc_nodes):
                distance = distances.get(int(target_node))
                if distance is None:
                    continue
                distance_buckets[row_idx, col_idx] = min(int(distance), 3)
        return torch.as_tensor(distance_buckets, dtype=torch.long, device=device)

    def _build_pair_matrices(
        self,
        sample: ContextGraphSample,
        candidate_local_idx: torch.Tensor,
        compat1_logits: torch.Tensor,
        compat2_logits: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        num_candidates = int(candidate_local_idx.numel())
        pair1_matrix = compat1_logits.new_zeros((num_candidates, num_candidates))
        pair2_matrix = compat2_logits.new_zeros((num_candidates, num_candidates))
        candidate_lookup = {int(local_idx): pos for pos, local_idx in enumerate(candidate_local_idx.tolist())}
        self._fill_pair_matrix(pair1_matrix, sample.pair1_index, compat1_logits, candidate_lookup)
        self._fill_pair_matrix(pair2_matrix, sample.pair2_index, compat2_logits, candidate_lookup)
        return pair1_matrix, pair2_matrix

    def _fill_pair_matrix(
        self,
        pair_matrix: torch.Tensor,
        pair_index: torch.Tensor,
        compat_logits: torch.Tensor,
        candidate_lookup: Dict[int, int],
    ) -> None:
        if pair_index.numel() == 0 or compat_logits.numel() == 0:
            return
        pair_probabilities = torch.sigmoid(compat_logits)
        pair_index_cpu = pair_index.detach().cpu().numpy()
        for pair_pos in range(pair_index_cpu.shape[1]):
            local_u = int(pair_index_cpu[0, pair_pos])
            local_v = int(pair_index_cpu[1, pair_pos])
            if local_u not in candidate_lookup or local_v not in candidate_lookup:
                continue
            idx_u = candidate_lookup[local_u]
            idx_v = candidate_lookup[local_v]
            value = pair_probabilities[pair_pos]
            pair_matrix[idx_u, idx_v] = value
            pair_matrix[idx_v, idx_u] = value

    def _singleton_scores(
        self,
        candidate_unary: torch.Tensor,
        coverage_matrix: torch.Tensor,
        count_log_probs: torch.Tensor,
    ) -> torch.Tensor:
        count_index = min(1, int(count_log_probs.numel()) - 1)
        return (
            self._unary_scale() * candidate_unary
            + self._coverage_scale() * coverage_matrix.mean(dim=1)
            + self._count_scale() * count_log_probs[count_index]
        )

    def _step_loss(self, context: dict) -> torch.Tensor:
        gold_mask = context["gold_mask"]
        gold_count = int(gold_mask.sum().item())
        if gold_count <= 0:
            return context["candidate_unary"].new_zeros(())

        selected_mask = torch.zeros_like(gold_mask, dtype=torch.bool)
        losses = []
        for _ in range(gold_count):
            remaining_gold = gold_mask & ~selected_mask
            if not bool(remaining_gold.any()):
                break
            gains = self._marginal_gains(selected_mask=selected_mask, context=context)
            logits = gains / max(float(self.config.step_temperature), 1e-6)
            target = remaining_gold.to(dtype=logits.dtype)
            target = target / target.sum().clamp_min(1.0)
            losses.append(-(target * F.log_softmax(logits, dim=0)).sum())

            masked_gains = gains.masked_fill(~remaining_gold, torch.finfo(gains.dtype).min)
            next_idx = int(torch.argmax(masked_gains).item())
            selected_mask[next_idx] = True

        if not losses:
            return context["candidate_unary"].new_zeros(())
        return torch.stack(losses).mean()

    def _set_loss(self, context: dict) -> torch.Tensor:
        gold_mask = context["gold_mask"]
        gold_count = int(gold_mask.sum().item())
        if gold_count <= 0:
            return context["candidate_unary"].new_zeros(())

        gold_score = self._score_mask(gold_mask, context)
        negatives: List[torch.Tensor] = []

        predicted_mask, _ = self._best_prefix(context)
        if not torch.equal(predicted_mask, gold_mask):
            negatives.append(predicted_mask)

        topk = min(gold_count, int(context["candidate_unary"].numel()))
        if topk > 0:
            unary_mask = torch.zeros_like(gold_mask)
            unary_mask[torch.topk(context["candidate_unary"], k=topk).indices] = True
            if not torch.equal(unary_mask, gold_mask):
                negatives.append(unary_mask)

            singleton_mask = torch.zeros_like(gold_mask)
            singleton_mask[torch.topk(context["singleton_scores"], k=topk).indices] = True
            if not torch.equal(singleton_mask, gold_mask):
                negatives.append(singleton_mask)

        if not negatives:
            return context["candidate_unary"].new_zeros(())

        losses = []
        for negative_mask in negatives:
            negative_score = self._score_mask(negative_mask, context)
            losses.append(torch.relu(self.config.set_margin - gold_score + negative_score))
        return torch.stack(losses).mean()

    def _best_prefix(self, context: dict) -> tuple[torch.Tensor, torch.Tensor]:
        candidate_count = int(context["candidate_unary"].numel())
        if candidate_count == 0:
            empty_mask = context["gold_mask"].new_zeros((0,), dtype=torch.bool)
            return empty_mask, context["candidate_unary"].new_zeros(())

        max_prefix = min(candidate_count, int(context["count_log_probs"].numel()) - 1)
        selected_mask = torch.zeros(candidate_count, dtype=torch.bool, device=context["candidate_unary"].device)
        prefix_masks = [selected_mask.clone()]
        prefix_scores = [self._score_mask(selected_mask, context)]

        for _ in range(max_prefix):
            gains = self._marginal_gains(selected_mask=selected_mask, context=context)
            next_idx = int(torch.argmax(gains).item())
            if bool(selected_mask[next_idx].item()):
                break
            selected_mask = selected_mask.clone()
            selected_mask[next_idx] = True
            prefix_masks.append(selected_mask.clone())
            prefix_scores.append(self._score_mask(selected_mask, context))

        score_tensor = torch.stack(prefix_scores)
        best_idx = int(torch.argmax(score_tensor).item())
        return prefix_masks[best_idx], score_tensor[best_idx]

    def _marginal_gains(self, selected_mask: torch.Tensor, context: dict) -> torch.Tensor:
        base_score = self._score_mask(selected_mask, context)
        gains = context["candidate_unary"].new_full((selected_mask.numel(),), torch.finfo(context["candidate_unary"].dtype).min)
        for candidate_idx in range(selected_mask.numel()):
            if bool(selected_mask[candidate_idx].item()):
                continue
            next_mask = selected_mask.clone()
            next_mask[candidate_idx] = True
            gains[candidate_idx] = self._score_mask(next_mask, context) - base_score
        return gains

    def _score_mask(self, selected_mask: torch.Tensor, context: dict) -> torch.Tensor:
        selected_indices = torch.nonzero(selected_mask, as_tuple=False).squeeze(-1)
        selected_count = int(selected_indices.numel())
        count_log_probs = context["count_log_probs"]
        count_term = self._count_scale() * count_log_probs[min(selected_count, int(count_log_probs.numel()) - 1)]
        if selected_count == 0:
            return count_term

        unary_term = self._unary_scale() * context["candidate_unary"][selected_indices].sum()
        coverage_term = self._coverage_scale() * context["coverage_matrix"][selected_indices].max(dim=0).values.mean()

        pair1_block = context["pair1_matrix"].index_select(0, selected_indices).index_select(1, selected_indices)
        pair2_block = context["pair2_matrix"].index_select(0, selected_indices).index_select(1, selected_indices)
        normalization = float(max(selected_count, 1))
        pair1_term = self._pair1_scale() * pair1_block.triu(diagonal=1).sum() / normalization
        pair2_term = self._pair2_scale() * pair2_block.triu(diagonal=1).sum() / normalization
        return unary_term + coverage_term + pair1_term + pair2_term + count_term

    @staticmethod
    def _count_log_probs(count_logits: torch.Tensor) -> torch.Tensor:
        if count_logits.numel() == 0:
            return count_logits.new_zeros((1,))
        continuation = torch.sigmoid(count_logits).clamp(min=1e-6, max=1.0 - 1e-6)
        prefix = torch.cat(
            [torch.ones(1, dtype=continuation.dtype, device=continuation.device), torch.cumprod(continuation, dim=0)[:-1]],
            dim=0,
        )
        count_probs = torch.cat(
            [prefix * (1.0 - continuation), torch.cumprod(continuation, dim=0)[-1:].clone()],
            dim=0,
        )
        count_probs = count_probs / count_probs.sum().clamp_min(1e-6)
        return torch.log(count_probs.clamp_min(1e-6))

    def _unary_scale(self) -> torch.Tensor:
        return F.softplus(self.unary_scale_raw) + 1e-6

    def _coverage_scale(self) -> torch.Tensor:
        return F.softplus(self.coverage_scale_raw) + 1e-6

    def _pair1_scale(self) -> torch.Tensor:
        return F.softplus(self.pair1_scale_raw) + 1e-6

    def _pair2_scale(self) -> torch.Tensor:
        return F.softplus(self.pair2_scale_raw) + 1e-6

    def _count_scale(self) -> torch.Tensor:
        return F.softplus(self.count_scale_raw) + 1e-6

    def _build_undirected_adjacency(
        self,
        pair1_index: torch.Tensor,
        maxcc_local_idx: torch.Tensor,
    ) -> Dict[int, List[int]]:
        adjacency_lookup = {int(node): [] for node in maxcc_local_idx.tolist()}
        pair1_index_np = pair1_index.detach().cpu().numpy()
        for edge_pos in range(pair1_index_np.shape[1]):
            local_u = int(pair1_index_np[0, edge_pos])
            local_v = int(pair1_index_np[1, edge_pos])
            adjacency_lookup.setdefault(local_u, []).append(local_v)
            adjacency_lookup.setdefault(local_v, []).append(local_u)
        return adjacency_lookup

    def _shortest_distances(self, source: int, adjacency_lookup: Dict[int, List[int]]) -> Dict[int, int]:
        distances: Dict[int, int] = {int(source): 0}
        queue: deque[int] = deque([int(source)])
        while queue:
            current = queue.popleft()
            next_distance = distances[current] + 1
            for neighbor in adjacency_lookup.get(current, []):
                neighbor = int(neighbor)
                if neighbor in distances:
                    continue
                distances[neighbor] = next_distance
                queue.append(neighbor)
        return distances
