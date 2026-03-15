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
    GFRR 损失函数: BCE + Ranking Loss + CC-Contrastive Loss
    
    Args:
        pos_weight: BCE 正样本权重 (默认 10.0)
        lambda_rank: Ranking Loss 权重 (默认 0.1)
        margin: Ranking Loss 边界 (默认 0.15)
        lambda_cc: CC-Contrastive Loss 权重 (默认 0.2)
        temperature: 对比学习温度参数 (默认 0.1)
    """
    def __init__(
        self,
        pos_weight: float = 10.0,
        lambda_rank: float = 0.1,
        margin: float = 0.15,
        lambda_cc: float = 0.2,
        temperature: float = 0.1
    ):
        super().__init__()
        
        self.lambda_rank = lambda_rank
        self.margin = margin
        self.lambda_cc = lambda_cc
        self.temperature = temperature
        
        # BCE Loss
        self.bce = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([pos_weight])
        )
    
    def forward(self, logits, target, mask, k_inf=None, z_cc_dict=None, cc_labels=None, max_cc_id=None):
        """
        计算总损失
        
        Args:
            logits: 分类 logits [N]
            target: 标签 [N]
            mask: 感染节点掩码 [N]
            k_inf: 感染邻居数 [N] (用于 Ranking Loss)
            z_cc_dict: CC级embedding字典 {cc_id: tensor} (用于 CC-Contrastive Loss)
            cc_labels: CC标签 [N] (用于 CC-Contrastive Loss)
            max_cc_id: 最大CC的ID (用于 CC-Contrastive Loss)
        
        Returns:
            loss_dict: 包含各损失项的字典
                - total: 总损失
                - cls: 分类损失
                - bce: BCE 损失
                - rank: Ranking 损失
                - cc_contrast: CC对比损失
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
        
        # 3. CC-Contrastive Loss
        if z_cc_dict is not None and cc_labels is not None and max_cc_id is not None:
            cc_contrast_loss = self._compute_cc_contrastive_loss(
                z_cc_dict, target, mask, cc_labels, max_cc_id
            )
        else:
            cc_contrast_loss = torch.tensor(0.0, device=device)
        
        # 总损失
        cls_loss = bce_loss + self.lambda_rank * rank_loss + self.lambda_cc * cc_contrast_loss
        
        return {
            'total': cls_loss,
            'cls': cls_loss,
            'bce': bce_loss,
            'rank': rank_loss,
            'cc_contrast': cc_contrast_loss
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
    
    def _compute_cc_contrastive_loss(self, z_cc_dict, target, mask, cc_labels, max_cc_id):
        """
        CC-Contrastive Loss: 拉近源点与最大CC，推远源点与孤岛
        
        策略:
            1. 找到所有源点节点
            2. 获取最大CC的embedding (正样本)
            3. 获取孤岛CC的embedding (负样本)
            4. 计算InfoNCE风格的对比损失
        
        Args:
            z_cc_dict: CC级embedding字典 {cc_id: tensor[D]}
            target: 标签 [N]
            mask: 感染节点掩码 [N]
            cc_labels: CC标签 [N]
            max_cc_id: 最大CC的ID
        
        Returns:
            cc_contrast_loss: CC对比损失
        """
        device = target.device
        
        if len(z_cc_dict) == 0:
            return torch.tensor(0.0, device=device)
        
        # 找到源点节点
        source_mask = (target == 1) & mask
        if source_mask.sum() == 0:
            return torch.tensor(0.0, device=device)
        
        source_indices = torch.where(source_mask)[0]
        
        # 获取最大CC的embedding
        if max_cc_id not in z_cc_dict:
            return torch.tensor(0.0, device=device)
        
        z_max_cc = z_cc_dict[max_cc_id]  # [D]
        
        # 收集孤岛CC的embedding (非最大CC)
        z_island_list = []
        for cc_id, z_cc in z_cc_dict.items():
            if cc_id != max_cc_id:
                z_island_list.append(z_cc)
        
        if len(z_island_list) == 0:
            # 没有孤岛，无需对比损失
            # 这是正常情况：中间快照或连通性好的级联
            return torch.tensor(0.0, device=device)
        
        z_islands = torch.stack(z_island_list)  # [n_islands, D]
        
        # 对每个源点计算对比损失
        total_loss = torch.tensor(0.0, device=device)
        num_sources = 0
        
        for src_idx in source_indices:
            # 获取源点所在CC
            src_cc_id = cc_labels[src_idx].item()
            if src_cc_id < 0 or src_cc_id not in z_cc_dict:
                continue
            
            z_source_cc = z_cc_dict[src_cc_id]  # [D]
            
            # 计算相似度 (cosine similarity)
            sim_positive = F.cosine_similarity(
                z_source_cc.unsqueeze(0), z_max_cc.unsqueeze(0)
            )  # [1]
            
            sim_negatives = F.cosine_similarity(
                z_source_cc.unsqueeze(0), z_islands, dim=1
            )  # [n_islands]
            
            # InfoNCE Loss
            exp_pos = torch.exp(sim_positive / self.temperature)
            exp_negs = torch.exp(sim_negatives / self.temperature).sum()
            
            loss = -torch.log(exp_pos / (exp_pos + exp_negs + 1e-8))
            total_loss = total_loss + loss
            num_sources += 1
        
        if num_sources > 0:
            total_loss = total_loss / num_sources
        
        return total_loss
