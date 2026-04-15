from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from config import ModelConfig
from .data_types import ContextGraphSample
from .gatv2 import StackedGATv2Encoder


@dataclass(frozen=True)
class ModelOutput:
    node_logits: torch.Tensor
    compat1_logits: torch.Tensor
    compat2_logits: torch.Tensor
    count_logits: torch.Tensor
    hidden_states: torch.Tensor


class PairHead(nn.Module):
    def __init__(self, hidden_dim: int, pair_feature_dim: int, output_dim: int, dropout: float) -> None:
        super().__init__()
        pair_input_dim = hidden_dim * 4 + pair_feature_dim
        self.output_dim = output_dim
        self.net = nn.Sequential(
            nn.Linear(pair_input_dim, hidden_dim),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, hidden_states: torch.Tensor, pair_index: torch.Tensor, pair_features: torch.Tensor) -> torch.Tensor:
        if pair_index.numel() == 0:
            return hidden_states.new_zeros((0, self.output_dim))
        src = hidden_states[pair_index[0]]
        dst = hidden_states[pair_index[1]]
        pair_input = torch.cat([src, dst, torch.abs(src - dst), src * dst, pair_features], dim=-1)
        return self.net(pair_input)


class CountHead(nn.Module):
    def __init__(self, hidden_dim: int, count_hidden_dim: int, max_count: int, dropout: float) -> None:
        super().__init__()
        if max_count <= 0:
            raise ValueError("max_count must be positive")
        self.pool_gate = nn.Sequential(
            nn.Linear(hidden_dim, count_hidden_dim),
            nn.ELU(),
            nn.Linear(count_hidden_dim, 1),
        )
        self.out_net = nn.Sequential(
            nn.Linear(hidden_dim, count_hidden_dim),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(count_hidden_dim, max_count),
        )

    def forward(self, hidden_states: torch.Tensor, maxcc_local_idx: torch.Tensor) -> torch.Tensor:
        maxcc_hidden = hidden_states[maxcc_local_idx]
        gate = torch.sigmoid(self.pool_gate(maxcc_hidden))
        pooled = (gate * maxcc_hidden).sum(dim=0) / gate.sum(dim=0).clamp_min(1e-6)
        return self.out_net(pooled)


class ContextRootModel(nn.Module):
    def __init__(self, num_node_features: int, pair_feature_dim: int, max_count: int, config: ModelConfig) -> None:
        super().__init__()
        self.encoder = StackedGATv2Encoder(
            in_dim=num_node_features,
            hidden_dim=config.hidden_dim,
            num_heads=config.num_heads,
            num_layers=config.num_layers,
            dropout=config.dropout,
            residual=config.residual,
        )
        self.root_head = nn.Sequential(
            nn.Linear(config.hidden_dim, config.root_hidden_dim),
            nn.ELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.root_hidden_dim, 1),
        )
        self.compat1_head = PairHead(
            hidden_dim=config.hidden_dim,
            pair_feature_dim=pair_feature_dim,
            output_dim=1,
            dropout=config.dropout,
        )
        self.compat2_head = PairHead(
            hidden_dim=config.hidden_dim,
            pair_feature_dim=pair_feature_dim,
            output_dim=1,
            dropout=config.dropout,
        )
        self.count_head = CountHead(
            hidden_dim=config.hidden_dim,
            count_hidden_dim=config.count_hidden_dim,
            max_count=max_count,
            dropout=config.dropout,
        )

    def forward(self, sample: ContextGraphSample) -> ModelOutput:
        hidden_states = self.encoder(sample.x, sample.edge_index_ctx)
        node_logits = self.root_head(hidden_states).squeeze(-1)
        compat1_logits = self.compat1_head(hidden_states, sample.pair1_index, sample.pair1_features).squeeze(-1)
        compat2_logits = self.compat2_head(hidden_states, sample.pair2_index, sample.pair2_features).squeeze(-1)
        count_logits = self.count_head(hidden_states, sample.maxcc_local_idx)
        return ModelOutput(
            node_logits=node_logits,
            compat1_logits=compat1_logits,
            compat2_logits=compat2_logits,
            count_logits=count_logits,
            hidden_states=hidden_states,
        )
