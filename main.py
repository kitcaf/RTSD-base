"""
GFRR 训练入口
使用 GFRRLite 模型 (Encoder + ClassHead)

使用方法:
    python main.py
"""
import os
import random
import numpy as np
import torch
from torch_geometric.loader import DataLoader

# 自定义模块
from config import (
    DEVICE, SEED, DATASETS, DATASET_NAMES, DATASET_IDX,
    TRAIN_RATIO, VAL_RATIO, RECALL_K_VALUES,
    LR, WEIGHT_DECAY, EPOCHS,
    USE_DYNAMIC_POS_WEIGHT, DEFAULT_POS_WEIGHT,
    HARD_GATE_MAX_CC, LOGIT_GATE_VALUE,
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
    apply_max_cc_hard_gate,
    setup_training_logger,
    log_print
)
from metrics_utils import precompute_shortest_paths


def train_epoch(model, loader, criterion, optimizer, device):
    """
    训练一个 epoch 
    """
    model.train()
    
    total_loss = 0
    loss_components = {'cls': 0, 'bce': 0, 'rank': 0}
    num_batches = 0
    
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        
        logits = model(data)
        if HARD_GATE_MAX_CC and hasattr(data, 'max_cc_mask'):
            logits = apply_max_cc_hard_gate(
                logits,
                train_mask=data.train_mask,
                max_cc_mask=data.max_cc_mask,
                gate_strength=LOGIT_GATE_VALUE
            )
        
        mask = data.train_mask
        if mask.sum() > 0:
            loss_dict = criterion(
                logits, data.y, mask,
                k_inf=data.k_inf if hasattr(data, 'k_inf') else None
            )
            
            loss = loss_dict['total']
            loss.backward()
            
            # 梯度裁剪 (与完整版一致)
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


def main():
    # 初始化
    setup_seed(SEED)
    
    # 获取数据集名称并配置日志
    data_name = DATASET_NAMES[DATASET_IDX]
    log_file_name = f"{data_name}_log.txt"
    logger = setup_training_logger(log_name=log_file_name)
    
    log_print(logger, "=" * 60)
    log_print(logger, "[*] 说明: 使用 GFRRLite (Encoder + ClassHead - 最大CC硬约束)")
    log_print(logger, "=" * 60)
    log_print(logger, f"[*] 设备: {DEVICE}")
    
    # 加载数据
    adj, influ = load_raw_data(DATASETS[DATASET_IDX])
    log_print(logger, f"[*] 数据集: {data_name}")
    log_print(logger, f"    节点数: {adj.shape[0]}, 级联数: {len(influ)}")
    
    # 获取配置
    arch_config = get_gfrr_arch_config()
    loss_config = get_gfrr_loss_config()
    
    # 特征工程
    engineer = FeatureEngineerGFRR(adj)
    dataset = engineer.generate_dataset(
        influ, 
        cache_name=data_name, 
        use_gfrr=False
    )
    
    # 预计算最短路径 (与完整版完全一致)
    log_print(logger, "[*] 预计算最短路径矩阵...")
    dist_matrix = precompute_shortest_paths(adj)
    
    # 数据划分 ── 按 cascade_id 分组，保证同一级联的不同快照落在同一分区
    # · train_set : 包含训练分区所有快照（含中间快照，扩充训练数据）
    # · val_set   : 只保留 is_final=True 快照（推理语义，与测试一致）
    # · test_set  : 只保留 is_final=True 快照
    all_cascade_ids = sorted(set(d.cascade_id for d in dataset))
    random.shuffle(all_cascade_ids)
    n_cas = len(all_cascade_ids)

    train_cas_ids = set(all_cascade_ids[:int(n_cas * TRAIN_RATIO)])
    val_cas_ids   = set(all_cascade_ids[int(n_cas * TRAIN_RATIO):int(n_cas * (TRAIN_RATIO + VAL_RATIO))])
    test_cas_ids  = set(all_cascade_ids[int(n_cas * (TRAIN_RATIO + VAL_RATIO)):])

    train_set = [d for d in dataset if d.cascade_id in train_cas_ids]
    val_set   = [d for d in dataset if d.cascade_id in val_cas_ids   and d.is_final]
    test_set  = [d for d in dataset if d.cascade_id in test_cas_ids  and d.is_final]

    log_print(logger, f"[*] 数据划分: Train={len(train_set)} (含所有快照), Val={len(val_set)} (仅最终快照), Test={len(test_set)} (仅最终快照)")
    log_print(logger, f"    梯度级联数: Train={len(train_cas_ids)}, Val={len(val_cas_ids)}, Test={len(test_cas_ids)}")

    
    train_loader = DataLoader(train_set, batch_size=1, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=1)
    test_loader = DataLoader(test_set, batch_size=1)
    
    # 确定 pos_weight (与完整版完全一致)
    if USE_DYNAMIC_POS_WEIGHT:
        pos_weight, _ = compute_dynamic_pos_weight(train_set)
        log_print(logger, f"[*] 动态 pos_weight: {pos_weight:.2f}")
    else:
        pos_weight = loss_config.get('pos_weight', DEFAULT_POS_WEIGHT)
        log_print(logger, f"[*] 静态 pos_weight: {pos_weight:.2f}")
    
    model = GFRRLite(
        num_features=14,
        hidden_dim=arch_config.get('hidden_dim', 32),
        encoder_blocks=arch_config.get('encoder_blocks', 3),
        dropout=arch_config.get('dropout', 0.3)
    ).to(DEVICE)
    
    # 打印模型参数
    param_info = model.get_num_params()
    log_print(logger, f"[*] 模型参数:")
    log_print(logger, f"    Encoder: {param_info['encoder']:,}")
    log_print(logger, f"    ClassHead: {param_info['class_head']:,}")
    log_print(logger, f"    Total: {param_info['total']:,}")
    
    criterion = GFRRLoss(
        pos_weight=pos_weight,
        lambda_rank=loss_config.get('lambda_rank', 0.1),
        margin=loss_config.get('margin', 0.15)
    ).to(DEVICE)
    
    # 优化器 (与完整版完全一致)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY
    )
    
    # 训练配置
    epochs = EPOCHS
    
    log_print(logger, f"\n{'='*60}")
    log_print(logger, f"      Epochs: {epochs}")
    log_print(logger, f"      LR: {LR}")
    log_print(logger, f"      Weight Decay: {WEIGHT_DECAY}")
    log_print(logger, f"      Lambda Rank: {loss_config.get('lambda_rank', 0.1)}")
    log_print(logger, f"      pos_weight: {pos_weight}")
    log_print(logger, f"      hidden_dim: {arch_config.get('hidden_dim', 32)}")
    log_print(logger, f"      encoder_blocks: {arch_config.get('encoder_blocks', 3)}")
    log_print(logger, f"      Max-CC硬门控: {HARD_GATE_MAX_CC} (gate={LOGIT_GATE_VALUE})")
    log_print(logger, f"      数据划分: cascade_id 分组 (Train=全快照, Val/Test=仅最终快照)")
    log_print(logger, f"{'='*60}\n")
    
    # 训练循环
    best_val_f1 = 0
    best_threshold = 0.5
    best_epoch = 0
    save_path = f'checkpoints_gfrr/gfrr_lite_{data_name}_ablation_noflow_best.pt'
    os.makedirs('checkpoints_gfrr', exist_ok=True)
    
    for epoch in range(epochs):
        avg_loss, loss_comp = train_epoch(model, train_loader, criterion, optimizer, DEVICE)
        
        best_th, _ = find_optimal_threshold_gfrr(
            model,
            val_loader,
            DEVICE,
            dist_matrix=dist_matrix,
            hard_gate_max_cc=HARD_GATE_MAX_CC,
            gate_strength=LOGIT_GATE_VALUE
        )
        val_metrics = evaluate_gfrr(
            model, val_loader, DEVICE,
            threshold=best_th,
            recall_k_values=RECALL_K_VALUES,
            dist_matrix=dist_matrix,
            hard_gate_max_cc=HARD_GATE_MAX_CC,
            gate_strength=LOGIT_GATE_VALUE
        )
        
        # 保存最佳模型
        if val_metrics['f1'] > best_val_f1:
            best_val_f1 = val_metrics['f1']
            best_threshold = best_th
            best_epoch = epoch + 1
            torch.save({
                'model_state_dict': model.state_dict(),
                'threshold': best_threshold,
                'epoch': best_epoch,
                'val_f1': best_val_f1
            }, save_path)
        
        # 打印进度
        recall_str = " | ".join([f"R@{k}: {val_metrics[f'recall@{k}']:.3f}" for k in RECALL_K_VALUES])
        log_print(logger, f"Epoch {epoch+1:03d} | Loss: {avg_loss:.4f} "
              f"(BCE:{loss_comp['bce']:.3f}, Rank:{loss_comp['rank']:.3f}) | "
              f"Val F1: {val_metrics['f1']:.4f} | {recall_str}")
    
    log_print(logger, f"[*] 最佳 Val F1: {best_val_f1:.4f} @ Epoch {best_epoch}")
    
    # 加载最佳模型并测试
    checkpoint = torch.load(save_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    test_metrics = evaluate_gfrr(
        model, test_loader, DEVICE,
        threshold=best_threshold,
        recall_k_values=RECALL_K_VALUES,
        dist_matrix=dist_matrix,
        hard_gate_max_cc=HARD_GATE_MAX_CC,
        gate_strength=LOGIT_GATE_VALUE
    )
    
    log_print(logger, f"\n{'='*60}")
    log_print(logger, f"[*] 测试结果:")
    log_print(logger, f"[*] 阈值: {best_threshold}")
    log_print(logger, f"    AUC       : {test_metrics['auc']:.4f}")
    log_print(logger, f"    Precision : {test_metrics['precision']:.4f}")
    log_print(logger, f"    Recall    : {test_metrics['recall']:.4f}")
    log_print(logger, f"    F1-Score  : {test_metrics['f1']:.4f}")
    for k in RECALL_K_VALUES:
        log_print(logger, f"    Recall@{k:<2} : {test_metrics[f'recall@{k}']:.4f}")
    log_print(logger, f"    MAP       : {test_metrics['map']:.4f}")
    log_print(logger, f"    P@K_true  : {test_metrics['p@k_true']:.4f}")
    log_print(logger, f"    AED       : {test_metrics['aed']:.4f}")
    log_print(logger, f"{'='*60}")

if __name__ == "__main__":
    main()
