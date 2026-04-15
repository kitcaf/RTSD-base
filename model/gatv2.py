from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class GATv2Layer(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        add_self_loops: bool = True,
    ) -> None:
        super().__init__()
        if in_dim <= 0 or out_dim <= 0 or num_heads <= 0:
            raise ValueError("in_dim, out_dim and num_heads must be positive")

        self.out_dim = out_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.add_self_loops = add_self_loops

        self.lin = nn.Linear(in_dim, out_dim * num_heads, bias=False)
        self.att = nn.Parameter(torch.empty(num_heads, out_dim))
        self.bias = nn.Parameter(torch.zeros(num_heads * out_dim))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.lin.weight)
        nn.init.xavier_uniform_(self.att)
        nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        if self.add_self_loops:
            num_nodes = x.size(0)
            loops = torch.arange(num_nodes, device=x.device, dtype=edge_index.dtype)
            self_loops = torch.stack([loops, loops], dim=0)
            edge_index = torch.cat([edge_index, self_loops], dim=1)

        src = edge_index[0]
        dst = edge_index[1]
        projected = self.lin(x).view(x.size(0), self.num_heads, self.out_dim)
        src_proj = projected[src]
        dst_proj = projected[dst]
        attention_input = F.leaky_relu(src_proj + dst_proj, negative_slope=0.2)
        attention_logits = (attention_input * self.att.unsqueeze(0)).sum(dim=-1)

        max_per_dst = torch.full(
            (x.size(0), self.num_heads),
            fill_value=torch.finfo(attention_logits.dtype).min,
            dtype=attention_logits.dtype,
            device=x.device,
        )
        max_per_dst.scatter_reduce_(
            0,
            dst.unsqueeze(-1).expand(-1, self.num_heads),
            attention_logits,
            reduce="amax",
            include_self=True,
        )
        normalized_logits = attention_logits - max_per_dst[dst]
        attention_weights = torch.exp(normalized_logits)
        denom = torch.zeros((x.size(0), self.num_heads), dtype=x.dtype, device=x.device)
        denom.index_add_(0, dst, attention_weights)
        attention_weights = attention_weights / denom[dst].clamp_min(1e-12)
        attention_weights = F.dropout(attention_weights, p=self.dropout, training=self.training)

        messages = src_proj * attention_weights.unsqueeze(-1)
        aggregated = torch.zeros((x.size(0), self.num_heads, self.out_dim), dtype=x.dtype, device=x.device)
        aggregated.index_add_(0, dst, messages)
        return aggregated.reshape(x.size(0), self.num_heads * self.out_dim) + self.bias


class StackedGATv2Encoder(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        num_heads: int,
        num_layers: int,
        dropout: float,
        residual: bool,
    ) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")
        self.dropout = dropout
        self.residual = residual

        self.input_proj = nn.Linear(in_dim, hidden_dim)
        head_dim = max(hidden_dim // num_heads, 1)
        self.layers = nn.ModuleList([
            GATv2Layer(
                in_dim=hidden_dim,
                out_dim=head_dim,
                num_heads=num_heads,
                dropout=dropout,
                add_self_loops=True,
            )
            for _ in range(num_layers)
        ])
        self.output_proj = nn.Linear(head_dim * num_heads, hidden_dim)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        hidden = self.input_proj(x)
        for layer in self.layers:
            residual = hidden
            hidden = layer(hidden, edge_index)
            hidden = self.output_proj(hidden)
            hidden = F.elu(hidden)
            if self.residual and residual.shape == hidden.shape:
                hidden = hidden + residual
            hidden = F.dropout(hidden, p=self.dropout, training=self.training)
        return hidden
