"""
特征消融实验 (Leave-One-Out Ablation Study)
每次移除一个特征，训练模型并记录性能

使用方法:
    python ablation_study.py
"""
import os
import random
import numpy as np
import torch
from torch_geometric.loader import DataLoader
from datetime import datetime

# 自定义模块
from config import (
    DEVICE, SEED, DATASETS, DATASET_NAMES, DATASET_IDX,
    TRAIN_RATIO, VAL_RATIO, RECALL_K_VALUES,
    LR, WEIGHT_DECAY, EPOCHS,
    USE_DYNAMIC_POS_WEIGHT, DEFAULT_POS_WEIGHT,
    get_gfrr_arch_config, get_gfrr_loss_config
)
from data_loader import load_raw_data
from feature_engineering_gfrr import FeatureEngineerGFRR
from model.gfrr import GFRRLite
from loss_gfrr import GFRRLoss
from utils import (
    setup_seed, 
    compute_dynamic_pos_weight,
    find_optimal_threshold_gfrr,
    evaluate_gfrr,
    log_print
)
from metrics_utils import precompute_shortest_paths


def train_epoch(model, loader, criterion, optimizer, device):
    """训练一个 epoch"""
    model.train()
    
    total_loss = 0
    loss_components = {'cls': 0, 'bce': 0, 'rank': 0, 'cc_contrast': 0}
    num_batches = 0
    
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        
        logits, z_cc_dict = model(data)
        
        mask = data.train_mask
        if mask.sum() > 0:
            is_final = data.is_final if hasattr(data, 'is_final') else True
            
            loss_dict = criterion(
                logits, data.y, mask,
                k_inf=data.k_inf if hasattr(data, 'k_inf') else None,
                z_cc_dict=z_cc_dict if (z_cc_dict and is_final) else None,
                cc_labels=data.cc_labels if (hasattr(data, 'cc_labels') and is_final) else None,
                max_cc_id=data.max_cc_id if (hasattr(data, 'max_cc_id') and is_final) else None
            )
            
            loss = loss_dict['total']
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += loss.item()
            for key in loss_components:
                if key in loss_dict:
                    val = loss_dict[key]
                    loss_components[key] += val.item() if torch.is_tensor(val) else val
            num_batches += 1
    
    avg_loss = total_loss / max(num_batches, 1)
    for key in loss_components:
        loss_components[key] /= max(num_batches, 1)
    
    return avg_loss, loss_components


def run_single_ablation(ablation_idx, feature_name, data_name, adj, influ, dist_matrix, result_file):
    """
    运行单次消融实验
    
    Args:
        ablation_idx: 要消融的特征索引
        feature_name: 特征名称
        data_name: 数据集名称
        adj: 邻接矩阵
        influ: 影响列表
        dist_matrix: 距离矩阵
        result_file: 结果文件句柄
    """
    setup_seed(SEED)
    
    log_msg = f"\n{'='*60}\n"
    log_msg += f"[*] 消融特征 [{ablation_idx}]: {feature_name}\n"
    log_msg += f"{'='*60}\n"
    print(log_msg)
    result_file.write(log_msg)
    result_file.flush()
    
    # 获取配置
    arch_config = get_gfrr_arch_config()
    loss_config = get_gfrr_loss_config()
    
    # 特征工程 (带消融参数)
    engineer = FeatureEngineerGFRR(adj, ablation_feature_idx=ablation_idx)
    
    # 生成数据集 (使用不同的缓存名)
    cache_suffix = f"_ablation_{ablation_idx}"
    dataset = engineer.generate_dataset(
        influ, 
        cache_name=f"{data_name}{cache_suffix}", 
        use_gfrr=False
    )
    
    # 数据划分
    all_cascade_ids = sorted(set(d.cascade_id for d in dataset))
    random.shuffle(all_cascade_ids)
    n_cas = len(all_cascade_ids)

    train_cas_ids = set(all_cascade_ids[:int(n_cas * TRAIN_RATIO)])
    val_cas_ids   = set(all_cascade_ids[int(n_cas * TRAIN_RATIO):int(n_cas * (TRAIN_RATIO + VAL_RATIO))])
    test_cas_ids  = set(all_cascade_ids[int(n_cas * (TRAIN_RATIO + VAL_RATIO)):])

    train_set = [d for d in dataset if d.cascade_id in train_cas_ids]
    val_set   = [d for d in dataset if d.cascade_id in val_cas_ids   and d.is_final]
    test_set  = [d for d in dataset if d.cascade_id in test_cas_ids  and d.is_final]
    
    train_loader = DataLoader(train_set, batch_size=1, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=1)
    test_loader = DataLoader(test_set, batch_size=1)
    
    # 确定 pos_weight
    if USE_DYNAMIC_POS_WEIGHT:
        pos_weight, _ = compute_dynamic_pos_weight(train_set)
    else:
        pos_weight = loss_config.get('pos_weight', DEFAULT_POS_WEIGHT)
    
    # 获取特征维度
    num_features = engineer.get_num_features()
    
    # 创建模型
    model = GFRRLite(
        num_features=num_features,
        hidden_dim=arch_config.get('hidden_dim', 32),
        encoder_blocks=arch_config.get('encoder_blocks', 3),
        dropout=arch_config.get('dropout', 0.3),
        beta=arch_config.get('beta', 1.0),
        lambda_1=arch_config.get('lambda_1', 0.5),
        lambda_2=arch_config.get('lambda_2', 1.0)
    ).to(DEVICE)
    
    criterion = GFRRLoss(
        pos_weight=pos_weight,
        lambda_rank=loss_config.get('lambda_rank', 0.1),
        margin=loss_config.get('margin', 0.15),
        lambda_cc=loss_config.get('lambda_cc', 0.2),
        temperature=loss_config.get('temperature', 0.1)
    ).to(DEVICE)
    
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY
    )
    
    # 训练循环
    best_val_f1 = 0
    best_threshold = 0.5
    best_epoch = 0
    epochs = EPOCHS
    
    for epoch in range(epochs):
        avg_loss, loss_comp = train_epoch(model, train_loader, criterion, optimizer, DEVICE)
        
        best_th, _ = find_optimal_threshold_gfrr(model, val_loader, DEVICE,
                                                  dist_matrix=dist_matrix)
        val_metrics = evaluate_gfrr(
            model, val_loader, DEVICE,
            threshold=best_th,
            recall_k_values=RECALL_K_VALUES, dist_matrix=dist_matrix
        )
        
        if val_metrics['f1'] > best_val_f1:
            best_val_f1 = val_metrics['f1']
            best_threshold = best_th
            best_epoch = epoch + 1
        
        # 每10个epoch打印一次 (仅控制台)
        if (epoch + 1) % 10 == 0:
            recall_str = " | ".join([f"R@{k}: {val_metrics[f'recall@{k}']:.3f}" for k in RECALL_K_VALUES])
            print(f"Epoch {epoch+1:03d} | Loss: {avg_loss:.4f} | Val F1: {val_metrics['f1']:.4f} | {recall_str}")
    
    # 测试
    test_metrics = evaluate_gfrr(
        model, test_loader, DEVICE,
        threshold=best_threshold,
        recall_k_values=RECALL_K_VALUES, dist_matrix=dist_matrix
    )
    
    # 记录结果 (与 main.py 格式完全一致)
    result_msg = f"\n{'='*60}\n"
    result_msg += f"[*] 测试结果:\n"
    result_msg += f"[*] 阈值: {best_threshold:.4f}\n"
    result_msg += f"    AUC       : {test_metrics['auc']:.4f}\n"
    result_msg += f"    Precision : {test_metrics['precision']:.4f}\n"
    result_msg += f"    Recall    : {test_metrics['recall']:.4f}\n"
    result_msg += f"    F1-Score  : {test_metrics['f1']:.4f}\n"
    for k in RECALL_K_VALUES:
        result_msg += f"    Recall@{k:<2} : {test_metrics[f'recall@{k}']:.4f}\n"
    result_msg += f"    MAP       : {test_metrics['map']:.4f}\n"
    result_msg += f"    P@K_true  : {test_metrics['p@k_true']:.4f}\n"
    result_msg += f"    AED       : {test_metrics['aed']:.4f}\n"
    result_msg += f"{'='*60}\n"
    
    print(result_msg)
    result_file.write(result_msg)
    result_file.flush()
    
    return test_metrics


def main():
    # 初始化
    setup_seed(SEED)
    data_name = DATASET_NAMES[DATASET_IDX]
    
    # 创建结果文件
    os.makedirs('history', exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_path = f'history/{data_name}_ablation_study_{timestamp}.txt'
    
    print(f"[*] 开始特征消融实验 (Leave-One-Out)")
    print(f"[*] 数据集: {data_name}")
    print(f"[*] 结果将保存到: {result_path}")
    
    # 加载数据
    adj, influ = load_raw_data(DATASETS[DATASET_IDX])
    print(f"[*] 节点数: {adj.shape[0]}, 级联数: {len(influ)}")
    
    # 预计算最短路径
    print("[*] 预计算最短路径矩阵...")
    dist_matrix = precompute_shortest_paths(adj)
    
    # 特征名称列表
    feature_names = FeatureEngineerGFRR.FEATURE_NAMES
    
    # 打开结果文件
    with open(result_path, 'w', encoding='utf-8') as f:
        header = f"""
{'='*80}
特征消融实验 (Leave-One-Out Ablation Study)
{'='*80}
数据集: {data_name}
节点数: {adj.shape[0]}
级联数: {len(influ)}
训练轮数: {EPOCHS}
学习率: {LR}
时间: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
{'='*80}

特征列表:
"""
        for idx, name in enumerate(feature_names):
            header += f"  [{idx:2d}] {name}\n"
        header += f"{'='*80}\n"
        
        print(header)
        f.write(header)
        f.flush()
        
        # 逐个消融特征 (共14次实验)
        for idx in range(14):
            print(f"\n[*] 步骤 {idx+1}/14: 消融特征 [{idx}] - {feature_names[idx]}")
            run_single_ablation(
                idx, feature_names[idx], data_name, adj, influ, dist_matrix, f
            )
    
    print(f"\n[*] 实验完成! 结果已保存到: {result_path}")


if __name__ == "__main__":
    main()
