"""
GFRR 损失函数模块

损失组成:
    L_total = L_cls = BCE + λ_rank * L_ranking
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from feature_engineering_gfrr import FeatureEngineerGFRR


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
        margin: float = 0.15,
        outside_bce_weight: float = 0.2,
        max_rank_positives: int = 4,
        hard_negative_topk: int = 8
    ):
        super().__init__()
        
        self.lambda_rank = lambda_rank
        self.margin = margin
        self.outside_bce_weight = outside_bce_weight
        self.max_rank_positives = max_rank_positives
        self.hard_negative_topk = hard_negative_topk
        self.feature_index = FeatureEngineerGFRR.FEATURE_INDEX
        
        # BCE Loss
        self.bce = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([pos_weight])
        )
    
    def forward(self, logits, target, mask, k_inf=None, context_mask=None, feature_bank=None, rank_mask=None):
        """
        计算总损失
        
        Args:
            logits: 分类 logits [N]
            target: 标签 [N]
            mask: 主监督掩码 [N]
            k_inf: 感染邻居数 [N] (用于 Ranking Loss)
            context_mask: 弱监督掩码 [N]
            feature_bank: 输入特征 [N, D]
            rank_mask: Ranking 专用掩码 [N]，为空时默认等于 mask
        
        Returns:
            loss_dict: 包含各损失项的字典
                - total: 总损失
                - cls: 分类损失
                - bce: BCE 损失
                - rank: Ranking 损失
        """
        device = logits.device
        mask = mask.bool()
        rank_mask = mask if rank_mask is None else (rank_mask.bool() & mask)
        if context_mask is not None:
            context_mask = context_mask.bool()
        
        # 确保 pos_weight 在正确设备
        if self.bce.pos_weight.device != device:
            self.bce.pos_weight = self.bce.pos_weight.to(device)
        
        # 1. BCE Loss (主监督 + 最大CC外弱监督)
        if mask.sum() > 0:
            bce_primary = self.bce(logits[mask], target[mask])
        else:
            bce_primary = torch.tensor(0.0, device=device)

        if context_mask is not None and context_mask.sum() > 0:
            bce_context = self.bce(logits[context_mask], target[context_mask])
        else:
            bce_context = torch.tensor(0.0, device=device)

        bce_loss = bce_primary + self.outside_bce_weight * bce_context
        
        # 2. Ranking Loss (在主监督区域内做精排)
        if k_inf is not None:
            rank_loss = self._compute_ranking_loss(
                logits=logits,
                target=target,
                k_inf=k_inf,
                mask=rank_mask,
                feature_bank=feature_bank
            )
        else:
            rank_loss = torch.tensor(0.0, device=device)
        
        # 总损失
        cls_loss = bce_loss + self.lambda_rank * rank_loss
        
        return {
            'total': cls_loss,
            'cls': cls_loss,
            'bce': bce_loss,
            'bce_primary': bce_primary,
            'bce_context': bce_context,
            'rank': rank_loss
        }
    
    def _gather_feature(self, feature_bank, indices, preferred_name, fallback_name=None):
        """从特征矩阵中读取一列，若不存在则回退到备用列。"""
        if feature_bank is None:
            return None

        if preferred_name in self.feature_index:
            pref_idx = self.feature_index[preferred_name]
            if pref_idx < feature_bank.size(1):
                return feature_bank[indices, pref_idx].float()

        if fallback_name is not None and fallback_name in self.feature_index:
            fallback_idx = self.feature_index[fallback_name]
            if fallback_idx < feature_bank.size(1):
                return feature_bank[indices, fallback_idx].float()

        return None

    def _compute_hard_negative_scores(self, logits, k_inf, feature_bank, neg_indices, candidate_mask):
        """为主监督区域内负样本构建结构感知难度分数。"""
        neg_probs = torch.sigmoid(logits[neg_indices])

        closeness = self._gather_feature(
            feature_bank, neg_indices,
            'closeness_percentile_in_maxcc',
            fallback_name='subgraph_closeness'
        )
        eccentricity = self._gather_feature(
            feature_bank, neg_indices,
            'eccentricity_percentile_in_maxcc',
            fallback_name='subgraph_eccentricity'
        )
        degree = self._gather_feature(
            feature_bank, neg_indices,
            'degree_percentile_in_maxcc',
            fallback_name='norm_deg'
        )
        kinf = self._gather_feature(
            feature_bank, neg_indices,
            'kinf_percentile_in_maxcc',
            fallback_name='norm_inf_count'
        )

        if kinf is None:
            masked_kinf = k_inf[candidate_mask].float()
            kinf_denom = masked_kinf.max().clamp_min(1.0)
            kinf = (k_inf[neg_indices].float() / kinf_denom).clamp(0.0, 1.0)

        if closeness is None:
            closeness = torch.zeros_like(neg_probs)
        if eccentricity is None:
            eccentricity = torch.zeros_like(neg_probs)
        if degree is None:
            degree = torch.zeros_like(neg_probs)

        structural_centrality = 0.5 * (closeness + eccentricity)
        hard_scores = (
            0.40 * neg_probs +
            0.25 * structural_centrality +
            0.15 * degree +
            0.20 * kinf
        )
        return hard_scores

    def _compute_ranking_loss(self, logits, target, k_inf, mask, feature_bank=None):
        """
        最大CC内部精排损失。

        策略:
            1. 仅在主监督区域内做 pairwise ranking
            2. 正样本优先选择当前打分偏低的难正样本
            3. 负样本优先选择当前高分、中心性高、度高、k_inf 高的 hardest negatives
        """
        device = logits.device
        mask = mask.bool()
        
        if mask.sum() == 0:
            return torch.tensor(0.0, device=device)
        
        logits_masked = logits[mask]
        target_masked = target[mask]
        
        source_mask = (target_masked == 1)
        non_source_mask = (target_masked == 0)
        
        num_sources = source_mask.sum().item()
        num_non_sources = non_source_mask.sum().item()
        
        if num_sources == 0 or num_non_sources == 0:
            return torch.tensor(0.0, device=device)
        
        global_indices = torch.where(mask)[0]
        pos_indices = global_indices[source_mask]
        neg_indices = global_indices[non_source_mask]

        pos_probs = torch.sigmoid(logits[pos_indices])
        num_pos = min(self.max_rank_positives, num_sources)
        hardest_pos = torch.topk(pos_probs, k=num_pos, largest=False).indices
        pos_scores = pos_probs[hardest_pos].unsqueeze(1)

        neg_hard_scores = self._compute_hard_negative_scores(
            logits=logits,
            k_inf=k_inf,
            feature_bank=feature_bank,
            neg_indices=neg_indices,
            candidate_mask=mask
        )
        num_neg = min(self.hard_negative_topk, num_non_sources)
        hardest_neg = torch.topk(neg_hard_scores, k=num_neg, largest=True).indices
        neg_scores = torch.sigmoid(logits[neg_indices[hardest_neg]]).unsqueeze(0)

        neg_weights = neg_hard_scores[hardest_neg]
        neg_weights = (neg_weights / neg_weights.mean().clamp_min(1e-6)).unsqueeze(0)

        pair_loss = F.relu(self.margin + neg_scores - pos_scores)
        pair_loss = pair_loss * neg_weights
        return pair_loss.mean()
    
