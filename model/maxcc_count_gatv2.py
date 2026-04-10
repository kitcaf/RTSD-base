"""
最大 CC 多任务 GATv2 模型

设计目标:
    1. 继续使用简单 2 层 GATv2 作为共享主干
    2. 并行输出节点源点评分与 max_cc 内源点个数预测
    3. 保持节点任务与 count 任务弱耦合，先稳定共享表示学习
"""
from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv


class MaxCCCountMultiTaskGATv2(nn.Module):
    """共享 GATv2 主干 + 节点打分头 + count head。"""

    GRAPH_STAT_DIM = 8

    def __init__(
        self,
        num_features: int,
        backbone_feature_dim: int,
        count_num_classes: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
        heads: int = 4,
        dropout: float = 0.3,
        count_hidden_dim: int = 32,
        count_dropout: float = 0.1,
    ) -> None:
        super().__init__()

        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")
        if backbone_feature_dim <= 0 or backbone_feature_dim > num_features:
            raise ValueError("backbone_feature_dim is out of valid range")
        if count_num_classes <= 1:
            raise ValueError("count_num_classes must be > 1")
        if count_hidden_dim <= 0:
            raise ValueError("count_hidden_dim must be positive")

        self.dropout = dropout
        self.backbone_feature_dim = backbone_feature_dim
        self.count_num_classes = count_num_classes

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

        self.node_classifier = nn.Linear(hidden_dim, 1)
        count_input_dim = hidden_dim * 2 + self.GRAPH_STAT_DIM
        self.count_head = nn.Sequential(
            nn.Linear(count_input_dim, count_hidden_dim),
            nn.ELU(),
            nn.Dropout(count_dropout),
            nn.Linear(count_hidden_dim, count_num_classes),
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

    def _get_batch_vector(self, data, num_nodes: int, device: torch.device) -> torch.Tensor:
        batch_vector = getattr(data, "batch", None)
        if batch_vector is None:
            return torch.zeros(num_nodes, dtype=torch.long, device=device)
        return batch_vector.to(device)

    def _get_count_mask(self, data, device: torch.device) -> torch.Tensor:
        infected_mask = getattr(data, "train_mask", None)
        if infected_mask is None:
            raise ValueError("data.train_mask is required for count estimation")

        count_mask = infected_mask.bool().to(device)
        max_cc_mask = getattr(data, "max_cc_mask", None)
        if max_cc_mask is not None:
            count_mask = count_mask & max_cc_mask.bool().to(device)
        return count_mask

    def _get_max_cc_ratio_tensor(self, data, num_graphs: int, device: torch.device) -> torch.Tensor:
        max_cc_ratio = getattr(data, "max_cc_ratio", None)
        if max_cc_ratio is None:
            return torch.zeros(num_graphs, 1, device=device)

        if torch.is_tensor(max_cc_ratio):
            ratio_tensor = max_cc_ratio.to(device=device, dtype=torch.float32).view(-1, 1)
        else:
            ratio_tensor = torch.tensor([[float(max_cc_ratio)]], device=device, dtype=torch.float32)

        if ratio_tensor.size(0) == 1 and num_graphs > 1:
            ratio_tensor = ratio_tensor.repeat(num_graphs, 1)
        if ratio_tensor.size(0) != num_graphs:
            raise ValueError("max_cc_ratio batch dimension does not match num_graphs")
        return ratio_tensor

    def _summarize_graph(
        self,
        hidden_states: torch.Tensor,
        node_logits: torch.Tensor,
        batch_vector: torch.Tensor,
        count_mask: torch.Tensor,
        infected_mask: torch.Tensor,
        data,
    ) -> torch.Tensor:
        num_graphs = int(batch_vector.max().item()) + 1 if batch_vector.numel() > 0 else 1
        device = hidden_states.device
        probs = torch.sigmoid(node_logits)
        max_cc_ratio = self._get_max_cc_ratio_tensor(data, num_graphs, device)

        graph_representations = []
        for graph_idx in range(num_graphs):
            graph_mask = batch_vector == graph_idx
            graph_count_mask = count_mask & graph_mask
            graph_infected_mask = infected_mask & graph_mask

            if graph_count_mask.any():
                graph_hidden = hidden_states[graph_count_mask]
                graph_scores = probs[graph_count_mask]
                mean_hidden = graph_hidden.mean(dim=0)
                max_hidden = graph_hidden.max(dim=0).values
            else:
                graph_scores = probs[graph_mask]
                mean_hidden = torch.zeros(hidden_states.size(-1), device=device)
                max_hidden = torch.zeros(hidden_states.size(-1), device=device)

            infected_count = float(graph_infected_mask.sum().item())
            max_cc_count = float(graph_count_mask.sum().item())

            if graph_scores.numel() > 0:
                sorted_scores, _ = torch.sort(graph_scores, descending=True)
                top_score = sorted_scores[0]
                second_score = sorted_scores[1] if sorted_scores.numel() > 1 else sorted_scores[0]
                top_gap = top_score - second_score
                mean_score = graph_scores.mean()
                std_score = graph_scores.std(unbiased=False) if graph_scores.numel() > 1 else torch.zeros(1, device=device).squeeze(0)
                clipped_scores = graph_scores.clamp(min=1e-6, max=1.0 - 1e-6)
                entropy = -(
                    clipped_scores * torch.log(clipped_scores) +
                    (1.0 - clipped_scores) * torch.log(1.0 - clipped_scores)
                ).mean()
            else:
                top_score = torch.zeros(1, device=device).squeeze(0)
                top_gap = torch.zeros(1, device=device).squeeze(0)
                mean_score = torch.zeros(1, device=device).squeeze(0)
                std_score = torch.zeros(1, device=device).squeeze(0)
                entropy = torch.zeros(1, device=device).squeeze(0)

            graph_stats = torch.stack(
                [
                    torch.log1p(torch.tensor(infected_count, device=device, dtype=torch.float32)),
                    torch.log1p(torch.tensor(max_cc_count, device=device, dtype=torch.float32)),
                    max_cc_ratio[graph_idx, 0],
                    mean_score,
                    top_score,
                    top_gap,
                    std_score,
                    entropy,
                ]
            )
            graph_representations.append(torch.cat([mean_hidden, max_hidden, graph_stats], dim=0))

        return torch.stack(graph_representations, dim=0)

    def forward(self, data) -> Dict[str, torch.Tensor]:
        backbone_x = data.x[:, :self.backbone_feature_dim]
        hidden_states = self.encode(backbone_x, data.edge_index)
        node_logits = self.node_classifier(hidden_states).squeeze(-1)

        batch_vector = self._get_batch_vector(data, backbone_x.size(0), backbone_x.device)
        infected_mask = data.train_mask.bool().to(backbone_x.device)
        count_mask = self._get_count_mask(data, backbone_x.device)
        graph_repr = self._summarize_graph(
            hidden_states=hidden_states,
            node_logits=node_logits,
            batch_vector=batch_vector,
            count_mask=count_mask,
            infected_mask=infected_mask,
            data=data,
        )
        count_logits = self.count_head(graph_repr)

        return {
            "node_logits": node_logits,
            "count_logits": count_logits,
        }
