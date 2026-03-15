"""
GFRR 损失函数模块

损失组成:
    L_backward = BCE + λ_rank * L_ranking (后向判别损失)
    L_forward = λ_focal * FocalLoss + λ_dice * DiceLoss (前向生成损失)
    L_total = L_backward + λ_forward * L_forward
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from loss_forward import ForwardLoss


class GFRRLoss(nn.Module):
    """
    GFRR 损失函数: BCE + Ranking Loss + (可选) Forward Loss
    
    Args:
        pos_weight: BCE 正样本权重 (默认 10.0)
        lambda_rank: Ranking Loss 权重 (默认 0.1)
        margin: Ranking Loss 边界 (默认 0.15)
        use_forward_loss: 是否启用前向损失 (默认 False)
        lambda_forward: 前向损失权重 (默认 0.5)
        lambda_focal: Focal Loss 权重 (默认 1.0)
        lambda_dice: Dice Loss 权重 (默认 0.5)
        focal_alpha: Focal Loss α (默认 0.25)
        focal_gamma: Focal Loss γ (默认 2.0)
    """
    def __init__(
        self,
        pos_weight: float = 10.0,
        lambda_rank: float = 0.1,
        margin: float = 0.15,
        use_forward_loss: bool = False,
        lambda_forward: float = 0.5,
        lambda_focal: float = 1.0,
        lambda_dice: float = 0.5,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0
    ):
        super().__init__()
        
        self.lambda_rank = lambda_rank
        self.margin = margin
        self.use_forward_loss = use_forward_loss
        self.lambda_forward = lambda_forward
        
        # BCE Loss
        self.bce = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([pos_weight])
        )
        
        # Forward Loss (可选)
        if use_forward_loss:
            self.forward_loss = ForwardLoss(
                lambda_focal=lambda_focal,
                lambda_dice=lambda_dice,
                focal_alpha=focal_alpha,
                focal_gamma=focal_gamma
            )
        else:
            self.forward_loss = None
    
    def forward(self, logits, target, mask, k_inf=None, O_pred=None, O_true_mask=None):
        """
        计算总损失
        
        Args:
            logits: 分类 logits [N] (后向判别)
            target: 源点标签 [N]
            mask: 感染节点掩码 [N]
            k_inf: 感染邻居数 [N] (用于 Ranking Loss)
            O_pred: 预测感染分布 [N] (前向生成, 可选)
            O_true_mask: 真实感染掩码 [N] (前向生成, 可选)
        
        Returns:
            loss_dict: 包含各损失项的字典
                - total: 总损失
                - backward: 后向判别损失
                - bce: BCE 损失
                - rank: Ranking 损失
                - forward: 前向生成损失 (可选)
                - focal: Focal Loss (可选)
                - dice: Dice Loss (可选)
        """
        device = logits.device
        
        # 确保 pos_weight 在正确设备
        if self.bce.pos_weight.device != device:
            self.bce.pos_weight = self.bce.pos_weight.to(device)
        
        # 1. 后向判别损失: BCE Loss
        if mask.sum() > 0:
            bce_loss = self.bce(logits[mask], target[mask])
        else:
            bce_loss = torch.tensor(0.0, device=device)
        
        # 2. 后向判别损失: Ranking Loss
        if k_inf is not None:
            rank_loss = self._compute_ranking_loss(logits, target, k_inf, mask)
        else:
            rank_loss = torch.tensor(0.0, device=device)
        
        # 后向总损失
        backward_loss = bce_loss + self.lambda_rank * rank_loss
        
        # 3. 前向生成损失 (可选)
        forward_loss_total = torch.tensor(0.0, device=device)
        focal_loss = torch.tensor(0.0, device=device)
        dice_loss = torch.tensor(0.0, device=device)
        
        if self.use_forward_loss and O_pred is not None and O_true_mask is not None:
            forward_dict = self.forward_loss(O_pred, O_true_mask)
            forward_loss_total = forward_dict['total']
            focal_loss = forward_dict['focal']
            dice_loss = forward_dict['dice']
        
        # 总损失
        total_loss = backward_loss + self.lambda_forward * forward_loss_total
        
        return {
            'total': total_loss,
            'backward': backward_loss,
            'bce': bce_loss,
            'rank': rank_loss,
            'forward': forward_loss_total,
            'focal': focal_loss,
            'dice': dice_loss
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
