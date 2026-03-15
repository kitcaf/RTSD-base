"""
前向生成器损失函数模块

损失组成:
    L_forward = λ_focal * FocalLoss + λ_dice * DiceLoss
    
设计原则:
    1. Masked 计算: 只在主连通块 (感染节点) 上计算损失
    2. Focal Loss: 处理类别不平衡, 聚焦难样本
    3. Dice Loss: 处理区域重叠, 优化 IoU
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class MaskedFocalLoss(nn.Module):
    """
    Masked Focal Loss
    
    公式:
        FL = -α * (1-p_t)^γ * log(p_t)
        p_t = p if y=1 else (1-p)
    
    Args:
        alpha: 正样本权重 (默认 0.25)
        gamma: 聚焦参数 (默认 2.0)
        reduction: 'mean' 或 'sum'
    """
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = 'mean'):
        super().__init__()
        
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, pred, target, mask):
        """
        计算 Focal Loss
        
        Args:
            pred: 预测概率 [N] (已经过 sigmoid)
            target: 真实标签 [N] (0 或 1)
            mask: 计算掩码 [N] (只在感染节点上计算)
        
        Returns:
            loss: Focal Loss 标量
        """
        if mask.sum() == 0:
            return torch.tensor(0.0, device=pred.device)
        
        # 只取掩码内的节点
        pred_masked = pred[mask]
        target_masked = target[mask]
        
        # 数值稳定性
        pred_masked = torch.clamp(pred_masked, min=1e-7, max=1.0 - 1e-7)
        
        # 计算 p_t
        p_t = torch.where(target_masked == 1, pred_masked, 1 - pred_masked)
        
        # Focal Loss
        focal_weight = (1 - p_t) ** self.gamma
        ce_loss = -torch.log(p_t)
        
        # 正负样本权重
        alpha_t = torch.where(target_masked == 1, 
                             torch.tensor(self.alpha, device=pred.device),
                             torch.tensor(1 - self.alpha, device=pred.device))
        
        loss = alpha_t * focal_weight * ce_loss
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class MaskedDiceLoss(nn.Module):
    """
    Masked Dice Loss
    
    公式:
        Dice = 1 - 2*|Ô∩O| / (|Ô|+|O|)
    
    优势: 直接优化 IoU, 对区域重叠敏感
    
    Args:
        smooth: 平滑项 (避免除零, 默认 1.0)
    """
    def __init__(self, smooth: float = 1.0):
        super().__init__()
        
        self.smooth = smooth
    
    def forward(self, pred, target, mask):
        """
        计算 Dice Loss
        
        Args:
            pred: 预测概率 [N] (已经过 sigmoid)
            target: 真实标签 [N] (0 或 1)
            mask: 计算掩码 [N] (只在感染节点上计算)
        
        Returns:
            loss: Dice Loss 标量
        """
        if mask.sum() == 0:
            return torch.tensor(0.0, device=pred.device)
        
        # 只取掩码内的节点
        pred_masked = pred[mask]
        target_masked = target[mask]
        
        # 计算交集和并集
        intersection = (pred_masked * target_masked).sum()
        pred_sum = pred_masked.sum()
        target_sum = target_masked.sum()
        
        # Dice 系数
        dice_coeff = (2.0 * intersection + self.smooth) / (pred_sum + target_sum + self.smooth)
        
        # Dice Loss
        loss = 1.0 - dice_coeff
        
        return loss


class ForwardLoss(nn.Module):
    """
    前向生成器总损失
    
    L_forward = λ_focal * FocalLoss + λ_dice * DiceLoss
    
    Args:
        lambda_focal: Focal Loss 权重 (默认 1.0)
        lambda_dice: Dice Loss 权重 (默认 0.5)
        focal_alpha: Focal Loss α (默认 0.25)
        focal_gamma: Focal Loss γ (默认 2.0)
        dice_smooth: Dice Loss 平滑项 (默认 1.0)
    """
    def __init__(
        self,
        lambda_focal: float = 1.0,
        lambda_dice: float = 0.5,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
        dice_smooth: float = 1.0
    ):
        super().__init__()
        
        self.lambda_focal = lambda_focal
        self.lambda_dice = lambda_dice
        
        self.focal_loss = MaskedFocalLoss(alpha=focal_alpha, gamma=focal_gamma)
        self.dice_loss = MaskedDiceLoss(smooth=dice_smooth)
    
    def forward(self, O_pred, O_true_mask):
        """
        计算前向损失
        
        Args:
            O_pred: 预测的感染分布 [N] (概率值)
            O_true_mask: 真实感染掩码 [N] (bool 或 0/1)
        
        Returns:
            loss_dict: 包含各损失项的字典
                - total: 总损失
                - focal: Focal Loss
                - dice: Dice Loss
        """
        device = O_pred.device
        
        # 转换为 float 标签
        O_true = O_true_mask.float()
        
        # 计算掩码 (只在感染节点上计算)
        mask = O_true_mask.bool()
        
        if mask.sum() == 0:
            return {
                'total': torch.tensor(0.0, device=device),
                'focal': torch.tensor(0.0, device=device),
                'dice': torch.tensor(0.0, device=device)
            }
        
        # Focal Loss
        focal = self.focal_loss(O_pred, O_true, mask)
        
        # Dice Loss
        dice = self.dice_loss(O_pred, O_true, mask)
        
        # 总损失
        total = self.lambda_focal * focal + self.lambda_dice * dice
        
        return {
            'total': total,
            'focal': focal,
            'dice': dice
        }
