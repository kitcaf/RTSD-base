"""
GFRR 主模型: Encoder + ClassificationHead + (可选) NeuralSIPropagator
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoder_gfrr import GFRREncoder
from .neural_si_propagator import NeuralSIPropagator


class ClassificationHead(nn.Module):
    """
    分类头: 将潜在表示解码为源点概率
    
    架构: MLP (D → D → 1)
    """
    def __init__(self, hidden_dim: int = 32, dropout: float = 0.3):
        super().__init__()
        
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, z):
        """
        Args:
            z: 潜在表示 [N, D]
        
        Returns:
            logits: 源点概率 logits [N]
        """
        return self.mlp(z).squeeze(-1)



class GFRRLite(nn.Module):
    """
    GFRR 模型: Encoder + ClassHead + (可选) Forward Propagator
    
    架构:
        后向判别: 观测态 → Encoder → ClassHead → 源点概率 P_S
        前向生成: P_S → NeuralSIPropagator → 感染分布 Ô (训练时)
    
    Args:
        use_forward_generator: 是否启用前向生成器 (默认 False)
        propagation_steps: 前向传播步数 (默认 3)
        beta_init: 初始感染烈度 (默认 1.0)
    """
    def __init__(
        self,
        num_features: int = 14,
        hidden_dim: int = 32,
        encoder_blocks: int = 3,
        dropout: float = 0.3,
        beta: float = 1.0,
        lambda_1: float = 0.5,
        lambda_2: float = 1.0,
        use_forward_generator: bool = False,
        propagation_steps: int = 3,
        beta_init: float = 1.0,
        infection_rate_hidden: int = 16
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.use_forward_generator = use_forward_generator
        
        # 后向判别器
        self.encoder = GFRREncoder(
            num_features=num_features,
            hidden_dim=hidden_dim,
            num_blocks=encoder_blocks,
            dropout=dropout,
            beta=beta,
            lambda_1=lambda_1,
            lambda_2=lambda_2
        )
        
        self.class_head = ClassificationHead(hidden_dim, dropout)
        
        # 前向生成器 (可选)
        if use_forward_generator:
            self.propagator = NeuralSIPropagator(
                k_steps=propagation_steps,
                beta_init=beta_init,
                infection_rate_hidden=infection_rate_hidden
            )
        else:
            self.propagator = None
    
    def forward(self, data, return_propagation=False):
        """
        前向传播
        
        Args:
            data: PyG Data 对象
            return_propagation: 是否返回前向生成结果 (训练时 True, 推理时 False)
        
        Returns:
            logits: 源点概率 logits [N]
            O_pred: 预测感染分布 [N] (仅当 return_propagation=True)
        """
        # 后向判别
        k_inf = data.k_inf if hasattr(data, 'k_inf') else None
        edge_dist = data.edge_dist if hasattr(data, 'edge_dist') else None
        degrees = data.degrees if hasattr(data, 'degrees') else None
        
        z, _ = self.encoder(data.x, data.edge_index, k_inf=k_inf, edge_dist=edge_dist, degrees=degrees)
        logits = self.class_head(z)
        
        # 前向生成 (仅训练时)
        if return_propagation and self.propagator is not None:
            P_S = torch.sigmoid(logits)  # 源点概率
            num_nodes = data.x.size(0)
            
            O_pred = self.propagator(
                P_S=P_S,
                edge_index=data.edge_index,
                degrees=degrees,
                num_nodes=num_nodes
            )
            
            return logits, O_pred
        
        return logits
    
    def get_num_params(self):
        """获取模型参数数量"""
        total = sum(p.numel() for p in self.parameters())
        encoder = sum(p.numel() for p in self.encoder.parameters())
        head = sum(p.numel() for p in self.class_head.parameters())
        
        result = {
            'total': total,
            'encoder': encoder,
            'class_head': head
        }
        
        if self.propagator is not None:
            propagator = sum(p.numel() for p in self.propagator.parameters())
            result['propagator'] = propagator
        
        return result
    
    def get_propagator_beta(self):
        """获取前向生成器的感染烈度"""
        if self.propagator is not None:
            return self.propagator.get_beta()
        return None
