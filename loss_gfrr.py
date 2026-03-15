"""
GFRR 损失函数模块

损失组成:
    L_total = L_cls = BCE + λ_rank * L_ranking
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class GFRRLoss(nn.Module):
    """
    GFRR 损失函数: BCE + Ranking Loss
    
    Args:
        pos_weight: BCE 正样本权重 (默认 10.0)
        lambda_rank: Ranking Loss 权重 (默认 0.1)
        margin: Ranking Loss 边界 (默认 0.15)
    """
    def __init__(
        self,
        pos_weight: float = 10.0,
        lambda_rank: float = 0.1,
        margin: float = 0.15
    ):
        super().__init__()
        
        self.lambda_rank = lambda_rank
        self.margin = margin
        
        # BCE Loss
        self.bce = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([pos_weight])
        )
    
    def forward(self, logits, target, mask, k_inf=None):
        """
        计算总损失
        
        Args:
            logits: 分类 logits [N]
            target: 标签 [N]
            mask: 感染节点掩码 [N]
            k_inf: 感染邻居数 [N] (用于 Ranking Loss)
        
        Returns:
            loss_dict: 包含各损失项的字典
                - total: 总损失
                - cls: 分类损失
                - bce: BCE 损失
                - rank: Ranking 损失
        """
        device = logits.device
        
        # 确保 pos_weight 在正确设备
        if self.bce.pos_weight.device != device:
            self.bce.pos_weight = self.bce.pos_weight.to(device)
        
        # 1. BCE Loss
        if mask.sum() > 0:
            bce_loss = self.bce(logits[mask], target[mask])
        else:
            bce_loss = torch.tensor(0.0, device=device)
        
        # 2. Ranking Loss
        if k_inf is not None:
            rank_loss = self._compute_ranking_loss(logits, target, k_inf, mask)
        else:
            rank_loss = torch.tensor(0.0, device=device)
        
        # 总损失
        cls_loss = bce_loss + self.lambda_rank * rank_loss
        
        return {
            'total': cls_loss,
            'cls': cls_loss,
            'bce': bce_loss,
            'rank': rank_loss,
        }
    
    def _compute_ranking_loss(self, logits, target, k_inf, mask):
        """
        Percentile-Aware Ranking Loss
        
        策略:
            1. 采样代表性源点 (中位数附近)
            2. 采样 k_inf 相似的 Hard Negative
            3. Hinge Loss: max(0, margin + score_neg - score_pos)
        """
        device = logits.device
        
        if mask.sum() == 0:
            return torch.tensor(0.0, device=device)
        
        logits_masked = logits[mask]
        target_masked = target[mask]
        k_inf_masked = k_inf[mask].float()
        
        source_mask = (target_masked == 1)
        non_source_mask = (target_masked == 0)
        
        num_sources = source_mask.sum().item()
        num_non_sources = non_source_mask.sum().item()
        
        if num_sources == 0 or num_non_sources == 0:
            return torch.tensor(0.0, device=device)
        
        # 采样源点
        source_indices = torch.where(source_mask)[0]
        source_kinf = k_inf_masked[source_indices]
        source_scores = logits_masked[source_indices]
        
        num_pos = min(3, num_sources)
        sorted_idx = torch.argsort(source_kinf)
        mid_start = max(0, (len(sorted_idx) - num_pos) // 2)
        selected_pos = sorted_idx[mid_start:mid_start + num_pos]
        
        # 采样负样本
        non_source_indices = torch.where(non_source_mask)[0]
        non_source_kinf = k_inf_masked[non_source_indices]
        non_source_scores = logits_masked[non_source_indices]
        
        total_loss = torch.tensor(0.0, device=device)
        num_pairs = 0
        
        for pos_local_idx in selected_pos:
            pos_kinf = source_kinf[pos_local_idx]
            pos_score = torch.sigmoid(source_scores[pos_local_idx])
            
            # 找 k_inf 相近的负样本
            kinf_diff = torch.abs(non_source_kinf - pos_kinf)
            num_neg = min(5, len(kinf_diff))
            _, closest_neg = torch.topk(kinf_diff, num_neg, largest=False)
            
            for neg_local_idx in closest_neg:
                neg_score = torch.sigmoid(non_source_scores[neg_local_idx])
                
                # Hinge Loss
                pair_loss = F.relu(self.margin + neg_score - pos_score)
                total_loss = total_loss + pair_loss
                num_pairs += 1
        
        if num_pairs > 0:
            total_loss = total_loss / num_pairs
        
        return total_loss
