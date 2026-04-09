"""
最大 CC 相对角色重打分模型

设计目标:
    1. 保留简单 2 层 GATv2 作为稳定主干
    2. 主干仅消费基础 + 最大CC相对特征，保持与阶段0可比
    3. 角色特征仅作为最大CC内部的 residual re-scoring 信号
"""
from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv


class MaxCCRoleRescoreHead(nn.Module):
    """对最大CC内部节点进行轻量角色残差重打分。"""

    def __init__(
        self,
        hidden_dim: int,
        role_feature_dim: int,
        hidden_width: int,
        dropout: float,
        residual_scale: float,
    ) -> None:
        super().__init__()

        if role_feature_dim <= 0:
            raise ValueError("role_feature_dim must be positive")
        if hidden_width <= 0:
            raise ValueError("hidden_width must be positive")

        self.residual_scale = residual_scale

        gate_input_dim = role_feature_dim + 1
        rescore_input_dim = hidden_dim + role_feature_dim + 1

        self.gate_net = nn.Sequential(
            nn.Linear(gate_input_dim, hidden_width),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_width, 1),
            nn.Sigmoid(),
        )
        self.delta_net = nn.Sequential(
            nn.Linear(rescore_input_dim, hidden_width),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_width, 1),
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        role_features: torch.Tensor,
        base_logits: torch.Tensor,
        max_cc_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        if max_cc_mask is None:
            return base_logits

        max_cc_mask = max_cc_mask.bool()
        if max_cc_mask.sum().item() == 0:
            return base_logits

        base_logit_col = base_logits.unsqueeze(-1)
        gate_input = torch.cat([role_features, base_logit_col], dim=-1)
        delta_input = torch.cat([hidden_states, role_features, base_logit_col], dim=-1)

        gate = self.gate_net(gate_input).squeeze(-1)
        delta = torch.tanh(self.delta_net(delta_input).squeeze(-1)) * self.residual_scale
        delta = delta * gate * max_cc_mask.float()
        return base_logits + delta


class MaxCCRelativeRoleGATv2(nn.Module):
    """简单 GATv2 主干 + 最大CC内部角色残差重打分。"""

    def __init__(
        self,
        num_features: int,
        backbone_feature_dim: int,
        role_feature_indices: Sequence[int],
        hidden_dim: int = 64,
        num_layers: int = 2,
        heads: int = 4,
        dropout: float = 0.3,
        role_hidden_dim: int = 32,
        role_dropout: float = 0.1,
        role_residual_scale: float = 1.0,
    ) -> None:
        super().__init__()

        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")
        if backbone_feature_dim <= 0 or backbone_feature_dim > num_features:
            raise ValueError("backbone_feature_dim is out of valid range")
        if not role_feature_indices:
            raise ValueError("role_feature_indices must not be empty")

        self.dropout = dropout
        self.backbone_feature_dim = backbone_feature_dim
        self.role_feature_indices = tuple(int(idx) for idx in role_feature_indices)

        self.convs = nn.ModuleList()
        in_dim = backbone_feature_dim
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

        self.base_classifier = nn.Linear(hidden_dim, 1)
        self.role_rescorer = MaxCCRoleRescoreHead(
            hidden_dim=hidden_dim,
            role_feature_dim=len(self.role_feature_indices),
            hidden_width=role_hidden_dim,
            dropout=role_dropout,
            residual_scale=role_residual_scale,
        )

    def encode(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        hidden_states = x
        for layer_idx, conv in enumerate(self.convs):
            residual = hidden_states
            hidden_states = conv(hidden_states, edge_index)
            hidden_states = F.elu(hidden_states)
            if layer_idx > 0 and hidden_states.shape == residual.shape:
                hidden_states = hidden_states + residual
            hidden_states = F.dropout(hidden_states, p=self.dropout, training=self.training)
        return hidden_states

    def forward(self, data) -> torch.Tensor:
        backbone_x = data.x[:, :self.backbone_feature_dim]
        role_x = data.x[:, self.role_feature_indices]

        hidden_states = self.encode(backbone_x, data.edge_index)
        base_logits = self.base_classifier(hidden_states).squeeze(-1)

        max_cc_mask = getattr(data, "max_cc_mask", None)
        return self.role_rescorer(
            hidden_states=hidden_states,
            role_features=role_x,
            base_logits=base_logits,
            max_cc_mask=max_cc_mask,
        )
