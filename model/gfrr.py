"""
GFRR 主模型: Encoder + ClassificationHead
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoder_gfrr import GFRREncoder


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
    GFRR 模型: Encoder + ClassHead
    """
    def __init__(
        self,
        num_features: int = 14,
        hidden_dim: int = 32,
        encoder_blocks: int = 3,
        dropout: float = 0.3,
        beta: float = 1.0,
        lambda_1: float = 0.5,
        lambda_2: float = 1.0
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        
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
    
    def forward(self, data):
        k_inf = data.k_inf if hasattr(data, 'k_inf') else None
        edge_dist = data.edge_dist if hasattr(data, 'edge_dist') else None
        degrees = data.degrees if hasattr(data, 'degrees') else None
        cc_labels = data.cc_labels if hasattr(data, 'cc_labels') else None
        train_mask = data.train_mask if hasattr(data, 'train_mask') else None
        
        z, z_cc_dict, _ = self.encoder(
            data.x, data.edge_index, 
            k_inf=k_inf, edge_dist=edge_dist, degrees=degrees,
            cc_labels=cc_labels, train_mask=train_mask
        )
        logits = self.class_head(z)
        return logits, z_cc_dict
    
    def get_num_params(self):
        """获取模型参数数量"""
        total = sum(p.numel() for p in self.parameters())
        encoder = sum(p.numel() for p in self.encoder.parameters())
        head = sum(p.numel() for p in self.class_head.parameters())
        return {
            'total': total,
            'encoder': encoder,
            'class_head': head
        }
