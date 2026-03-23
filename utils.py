"""
工具函数模块
"""
import os
import random
import numpy as np
import torch
import logging
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

def setup_seed(seed=42):
    """设置随机种子"""
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def setup_training_logger(log_dir='history', log_name='training_log.txt'):
    """
    配置日志记录器，保存到 history 文件夹，每次运行清空日志。
    """
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, log_name)
    
    logger = logging.getLogger('gfrr_training')
    logger.setLevel(logging.INFO)
    
    # 清除之前的handlers，防止重复
    if logger.hasHandlers():
        logger.handlers.clear()
        
    # File Handler (mode='w' 覆盖模式，每次运行清空)
    file_handler = logging.FileHandler(log_path, mode='w', encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    
    # Formatter
    formatter = logging.Formatter('%(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    file_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)
    
    return logger


def log_print(logger, message):
    """同时打印到控制台和记录到日志"""
    print(message)
    if logger:
        logger.info(message)


def compute_dynamic_pos_weight(dataset, scale_factor=1.0, min_weight=2.0, max_weight=50.0):
    """
    根据数据集的实际正负样本比例动态计算 pos_weight
    
    理论基础：Class-Balanced Loss (CVPR 2019)
    公式: pos_weight = (1 - 源点比例) / 源点比例 × scale_factor
    
    Args:
        dataset: PyG Data 列表
        scale_factor: 缩放因子
        min_weight: 最小权重
        max_weight: 最大权重
    
    Returns:
        pos_weight: 动态计算的正样本权重
        stats: 统计信息字典
    """
    total_pos = 0
    total_neg = 0
    total_infected = 0
    
    for data in dataset:
        mask = data.train_mask
        y = data.y
        
        infected_labels = y[mask]
        num_pos = (infected_labels == 1).sum().item()
        num_neg = (infected_labels == 0).sum().item()
        
        total_pos += num_pos
        total_neg += num_neg
        total_infected += mask.sum().item()
    
    if total_infected == 0:
        return 10.0, {'error': 'No infected nodes'}
    
    pos_ratio = total_pos / total_infected
    neg_ratio = total_neg / total_infected
    
    if pos_ratio > 0:
        raw_weight = neg_ratio / pos_ratio * scale_factor
    else:
        raw_weight = max_weight
    
    pos_weight = np.clip(raw_weight, min_weight, max_weight)
    
    stats = {
        'total_cascades': len(dataset),
        'total_infected': total_infected,
        'total_sources': total_pos,
        'total_non_sources': total_neg,
        'pos_ratio': pos_ratio,
        'raw_weight': raw_weight,
        'final_weight': pos_weight
    }
    
    return pos_weight, stats


def apply_max_cc_hard_gate(logits, train_mask=None, max_cc_mask=None, gate_strength=12.0):
    """
    对最大CC外感染节点执行输出层硬门控（logit下压）。

    Args:
        logits: [N] 原始logits
        train_mask: [N] 感染掩码，若提供则仅对感染节点施加门控
        max_cc_mask: [N] 最大CC掩码（True表示在最大CC内）
        gate_strength: logit下压值

    Returns:
        gated_logits: [N] 门控后的logits
    """
    if max_cc_mask is None:
        return logits

    max_cc_mask = max_cc_mask.bool().to(logits.device)
    outside_mask = ~max_cc_mask
    if train_mask is not None:
        outside_mask = outside_mask & train_mask.bool().to(logits.device)

    if outside_mask.sum() == 0:
        return logits

    gated_logits = logits.clone()
    gated_logits[outside_mask] = gated_logits[outside_mask] - gate_strength
    return gated_logits


# ==========================================
# GFRR 专用评估函数
# ==========================================

def find_optimal_threshold_gfrr(model, loader, device, dist_matrix=None,
                                hard_gate_max_cc=False, gate_strength=12.0):
    """
    GFRR 模型的 Per-Cascade 动态阈值搜索
    
    Args:
        model: GFRR 模型
        loader: 数据加载器
        device: 设备
    
    Returns:
        best_th: 最佳阈值
        best_f1: 最佳 F1 分数
    """
    model.eval()
    thresholds = np.arange(0.1, 0.95, 0.01)
    
    # 收集每个级联的预测结果
    cascade_data = []
    
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            
            # GFRR 使用 forward (返回logits和z_cc_dict)
            output = model(data)
            if isinstance(output, tuple):
                logits, _ = output
            else:
                logits = output

            if hard_gate_max_cc and hasattr(data, 'max_cc_mask'):
                logits = apply_max_cc_hard_gate(
                    logits,
                    train_mask=data.train_mask,
                    max_cc_mask=data.max_cc_mask,
                    gate_strength=gate_strength
                )
            probs = torch.sigmoid(logits)
            
            mask = data.train_mask
            if mask.sum() > 0:
                y_true = data.y[mask].cpu().numpy()
                y_scores = probs[mask].cpu().numpy()
                cascade_data.append((y_true, y_scores))
    
    if len(cascade_data) == 0:
        return 0.5, 0.0
    
    # Per-Cascade 阈值搜索
    best_f1 = 0
    best_th = 0.5
    
    for th in thresholds:
        f1_list = []
        for y_true, y_scores in cascade_data:
            y_pred = (y_scores > th).astype(int)
            # 确保至少预测一个节点
            if y_pred.sum() == 0 and len(y_scores) > 0:
                y_pred[np.argmax(y_scores)] = 1
            f1_list.append(f1_score(y_true, y_pred, zero_division=0))
        
        avg_f1 = np.mean(f1_list)
        if avg_f1 > best_f1:
            best_f1 = avg_f1
            best_th = th
    
    return best_th, best_f1


def evaluate_gfrr(model, loader, device, threshold=0.5,
                  recall_k_values=None, dist_matrix=None,
                  use_mc_dropout=False, mc_dropout_samples=10,
                  node_indices=None,
                  hard_gate_max_cc=False, gate_strength=12.0):
    """
    GFRR 模型评估函数
    
    Args:
        model: GFRR 模型
        loader: 数据加载器
        device: 设备
        threshold: 分类阈值 (建议使用 find_optimal_threshold_gfrr 获取)
        recall_k_values: Recall@K 的 K 值列表
        dist_matrix: 最短路径矩阵 (用于 AED)
        use_mc_dropout: 是否使用 MC Dropout 进行不确定性采样 (默认 False)
        mc_dropout_samples: MC Dropout 采样次数 (默认 10)
    
    Returns:
        metrics_dict: 评估指标字典，包含:
            - auc, precision, recall, f1
            - recall@k (各 K 值)
            - map, p@k_true, aed
    """
    from metrics_utils import calculate_map, calculate_precision_at_k, calculate_aed
    
    model.eval()
    
    # 定义启用 Dropout 的函数 (用于 MC Dropout)
    def enable_dropout(m):
        if type(m) == torch.nn.Dropout:
            m.train()
    
    if recall_k_values is None:
        recall_k_values = [5, 15, 25]
    
    # 指标收集列表
    precision_list, recall_list, f1_list, auc_list = [], [], [], []
    recall_at_k_lists = {k: [] for k in recall_k_values}
    map_list, pk_list, aed_list = [], [], []
    
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            
            # GFRR 推理
            if use_mc_dropout:
                # --- MC Dropout 采样逻辑 ---
                # 1. 开启 Dropout (保持其他层如 BN 为 eval 模式)
                model.apply(enable_dropout)
                
                logits_sum = None
                for _ in range(mc_dropout_samples):
                    output = model(data)
                    if isinstance(output, tuple):
                        curr_logits, _ = output
                    else:
                        curr_logits = output
                    
                    if logits_sum is None:
                        logits_sum = curr_logits
                    else:
                        logits_sum += curr_logits
                
                logits = logits_sum / mc_dropout_samples

                if hard_gate_max_cc and hasattr(data, 'max_cc_mask'):
                    logits = apply_max_cc_hard_gate(
                        logits,
                        train_mask=data.train_mask,
                        max_cc_mask=data.max_cc_mask,
                        gate_strength=gate_strength
                    )
                
                # 2. 恢复 eval 模式 (关闭 Dropout)
                model.eval()
            else:
                # --- 标准确定性推理 ---
                output = model(data)
                if isinstance(output, tuple):
                    logits, _ = output
                else:
                    logits = output

                if hard_gate_max_cc and hasattr(data, 'max_cc_mask'):
                    logits = apply_max_cc_hard_gate(
                        logits,
                        train_mask=data.train_mask,
                        max_cc_mask=data.max_cc_mask,
                        gate_strength=gate_strength
                    )

            probs = torch.sigmoid(logits)
            
            mask = data.train_mask
            if mask.sum() == 0:
                continue
            
            y_true = data.y[mask].cpu().numpy()
            y_scores = probs[mask].cpu().numpy()

            # 全局索引 (用于 AED)
            if hasattr(data, 'orig_node_ids'):
                local_mask = mask.cpu().numpy()
                infected_indices = data.orig_node_ids.cpu().numpy()[local_mask]
            elif node_indices is not None:
                infected_indices = np.asarray(node_indices)
            else:
                infected_indices = torch.where(mask)[0].cpu().numpy()
            
            # AUC
            try:
                if len(np.unique(y_true)) > 1:
                    auc_list.append(roc_auc_score(y_true, y_scores))
            except:
                pass
            
            # 源点信息
            num_sources = int(y_true.sum())
            num_infected = len(y_scores)
            
            if num_sources > 0:
                # Recall@K
                sorted_idx = np.argsort(-y_scores)
                for k in recall_k_values:
                    top_k = sorted_idx[:k]
                    hits = y_true[top_k].sum()
                    recall_at_k_lists[k].append(hits / num_sources)
                
                # MAP
                map_score = calculate_map(y_scores, y_true, num_infected)
                map_list.append(map_score)
                
                # P@K_true
                pk_score = calculate_precision_at_k(y_scores, y_true, num_sources)
                pk_list.append(pk_score)
                
                # AED
                if dist_matrix is not None:
                    aed_score = calculate_aed(
                        y_scores, y_true,
                        dist_matrix=dist_matrix,
                        top_k=num_sources,
                        node_indices=infected_indices
                    )
                    aed_list.append(aed_score)
            
            # Precision/Recall/F1 (基于阈值)
            y_pred = (y_scores > threshold).astype(int)
            if y_pred.sum() == 0 and len(y_scores) > 0:
                y_pred[np.argmax(y_scores)] = 1
            
            precision_list.append(precision_score(y_true, y_pred, zero_division=0))
            recall_list.append(recall_score(y_true, y_pred, zero_division=0))
            f1_list.append(f1_score(y_true, y_pred, zero_division=0))
    
    # 汇总指标
    metrics = {
        'auc': np.mean(auc_list) if auc_list else 0.0,
        'precision': np.mean(precision_list) if precision_list else 0.0,
        'recall': np.mean(recall_list) if recall_list else 0.0,
        'f1': np.mean(f1_list) if f1_list else 0.0,
        'map': np.mean(map_list) if map_list else 0.0,
        'p@k_true': np.mean(pk_list) if pk_list else 0.0,
        'aed': np.mean(aed_list) if aed_list else 0.0
    }
    
    for k in recall_k_values:
        metrics[f'recall@{k}'] = np.mean(recall_at_k_lists[k]) if recall_at_k_lists[k] else 0.0
    
    return metrics
