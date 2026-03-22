"""
GFRR 主模型: Encoder + ClassificationHead
"""
import torch
import torch.nn as nn

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
        use_max_cc_pool: bool = True,
        max_cc_pool_use_mlp: bool = True,
        max_cc_pool_alpha: float = 0.2,
        max_cc_pool_outside_alpha: float = 0.0
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        
        self.encoder = GFRREncoder(
            num_features=num_features,
            hidden_dim=hidden_dim,
            num_blocks=encoder_blocks,
            dropout=dropout,
            use_max_cc_pool=use_max_cc_pool,
            max_cc_pool_use_mlp=max_cc_pool_use_mlp,
            max_cc_pool_alpha=max_cc_pool_alpha,
            max_cc_pool_outside_alpha=max_cc_pool_outside_alpha
        )
        
        self.class_head = ClassificationHead(hidden_dim, dropout)
    
    def forward(self, data):
        max_cc_mask = data.max_cc_mask if hasattr(data, 'max_cc_mask') else None
        z, gate_weights = self.encoder(data.x, data.edge_index, max_cc_mask=max_cc_mask)
        logits = self.class_head(z)
        return logits
    
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
