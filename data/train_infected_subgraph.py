"""
使用感染子图（较大连通分量）进行 GFRR 训练
按照要求：
1. 传入图为含有较大感染节点的连通块（提取最大连通分量）
2. 提前筛选源点：源点只能在这个最大的连通块中
3. 训练使用：只使用最终快照状态和初始快照状态，不使用中间状态
"""
import sys
import os
import random
import numpy as np
import networkx as nx
import torch
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data

# 将上级目录加入路径，以便复用现有模块
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

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
    setup_training_logger,
    log_print
)
from metrics_utils import precompute_shortest_paths
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from metrics_utils import calculate_map, calculate_precision_at_k, calculate_aed

def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    loss_components = {'cls': 0, 'bce': 0, 'rank': 0}
    num_batches = 0
    
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        logits = model(data)
        
        mask = data.train_mask
        if mask.sum() > 0:
            loss_dict = criterion(
                logits, data.y, mask,
                k_inf=data.k_inf if hasattr(data, 'k_inf') else None
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


# 这里重写评估函数以支持动态的 dist_matrix，因为每个级联的图大小都不同
def find_optimal_threshold_subgraph(model, loader, device):
    model.eval()
    thresholds = np.arange(0.1, 0.95, 0.01)
    cascade_data = []
    
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            logits = model(data)
            probs = torch.sigmoid(logits)
            mask = data.train_mask
            if mask.sum() > 0:
                y_true = data.y[mask].cpu().numpy()
                y_scores = probs[mask].cpu().numpy()
                cascade_data.append((y_true, y_scores))
                
    if len(cascade_data) == 0: return 0.5, 0.0
    
    best_f1, best_th = 0, 0.5
    for th in thresholds:
        f1_list = []
        for y_true, y_scores in cascade_data:
            y_pred = (y_scores > th).astype(int)
            if y_pred.sum() == 0 and len(y_scores) > 0:
                y_pred[np.argmax(y_scores)] = 1
            f1_list.append(f1_score(y_true, y_pred, zero_division=0))
        avg_f1 = np.mean(f1_list)
        if avg_f1 > best_f1:
            best_f1 = avg_f1
            best_th = th
    return best_th, best_f1

def evaluate_subgraph(model, loader, device, threshold=0.5, recall_k_values=None, dist_matrix=None):
    model.eval()
    if recall_k_values is None:
        recall_k_values = [5, 15, 25]
        
    precision_list, recall_list, f1_list, auc_list = [], [], [], []
    recall_at_k_lists = {k: [] for k in recall_k_values}
    map_list, pk_list, aed_list = [], [], []
    
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            logits = model(data)
            probs = torch.sigmoid(logits)
            
            mask = data.train_mask
            if mask.sum() == 0: continue
            
            y_true = data.y[mask].cpu().numpy()
            y_scores = probs[mask].cpu().numpy()
            
            try:
                if len(np.unique(y_true)) > 1:
                    auc_list.append(roc_auc_score(y_true, y_scores))
            except: pass
            
            # 全局索引 (用于 AED)
            infected_indices = torch.where(mask)[0].cpu().numpy()
            global_node_ids = data.global_node_ids.cpu().numpy() if hasattr(data, 'global_node_ids') else None
            
            num_sources = int(y_true.sum())
            if num_sources > 0:
                sorted_idx = np.argsort(-y_scores)
                for k in recall_k_values:
                    top_k = sorted_idx[:k]
                    hits = y_true[top_k].sum()
                    recall_at_k_lists[k].append(hits / num_sources)
                    
                map_list.append(calculate_map(y_scores, y_true, len(y_scores)))
                pk_list.append(calculate_precision_at_k(y_scores, y_true, num_sources))
                
                # AED: 通过 global_node_ids 映射回原始网络计算距离
                if dist_matrix is not None and global_node_ids is not None:
                    node_indices = global_node_ids[infected_indices]
                    aed_score = calculate_aed(
                        y_scores, y_true,
                        dist_matrix=dist_matrix,
                        top_k=num_sources,
                        node_indices=node_indices
                    )
                    aed_list.append(aed_score)
            
            y_pred = (y_scores > threshold).astype(int)
            if y_pred.sum() == 0 and len(y_scores) > 0:
                y_pred[np.argmax(y_scores)] = 1
                
            precision_list.append(precision_score(y_true, y_pred, zero_division=0))
            recall_list.append(recall_score(y_true, y_pred, zero_division=0))
            f1_list.append(f1_score(y_true, y_pred, zero_division=0))
            
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


def main():
    setup_seed(SEED)
    data_name = DATASET_NAMES[DATASET_IDX]
    log_file_name = f"{data_name}_subgraph_log.txt"
    log_dir = os.path.join(os.path.dirname(__file__), '..', 'history')
    logger = setup_training_logger(log_dir=log_dir, log_name=log_file_name)
    
    log_print(logger, "=" * 60)
    log_print(logger, "[*] 说明: 使用 GFRRLite + 感染最大连通子图进行训练 (只使用初始/最终快照)")
    log_print(logger, "=" * 60)
    log_print(logger, f"[*] 设备: {DEVICE}")
    
    adj, influ = load_raw_data(DATASETS[DATASET_IDX])
    log_print(logger, f"[*] 数据集: {data_name}")
    log_print(logger, f"    全局节点数: {adj.shape[0]}, 总级联数: {len(influ)}")
    
    arch_config = get_gfrr_arch_config()
    loss_config = get_gfrr_loss_config()
    
    # 预计算全局最短路径矩阵 (用于 AED 指标)
    log_print(logger, "[*] 预计算全局最短路径矩阵...")
    dist_matrix = precompute_shortest_paths(adj)
    
    G_global = nx.from_numpy_array(adj)
    dataset = []
    
    log_print(logger, "[*] 正在构建基于感染子图的数据集...")
    for cascade_idx, mat in enumerate(influ):
        source_vec = mat[:, 0]
        final_infected = mat[:, -1]
        n_snapshot_cols = mat.shape[1] - 1
        
        source_nodes = set(np.where(source_vec == 1)[0])
        all_infected = set(np.where(final_infected == 1)[0]) | source_nodes
        
        if len(all_infected) < 2:
            continue
            
        G_sub_full = G_global.subgraph(list(all_infected))
        components = list(nx.connected_components(G_sub_full))
        if not components:
            continue
            
        # 1. 取最大的连通块（即大的联通块）
        largest_cc = max(components, key=len)
        
        # 2. 初步的源点筛选：筛选在这个连通块中的源点
        valid_sources = source_nodes & largest_cc
        if not valid_sources:
            # 如果最大的感染特征块里都没有源点，放弃该级联
            continue
            
        # 获取子图节点并重映射 ID
        cc_nodes = sorted(list(largest_cc))
        node2idx = {old_id: new_id for new_id, old_id in enumerate(cc_nodes)}
        N_sub = len(cc_nodes)
        
        G_cc = G_global.subgraph(cc_nodes)
        adj_sub = nx.to_numpy_array(G_cc, nodelist=cc_nodes)
        
        # 使用当前级联的子图去实例化 FeatureEngineer
        engineer = FeatureEngineerGFRR(adj_sub)
        
        # 直接使用真实图边（不构建 K-hop 虚拟图）
        rows, cols = np.where(adj_sub > 0)
        edge_index = torch.LongTensor(np.array([rows, cols]))
        edge_dist = torch.ones(edge_index.size(1), dtype=torch.float32)
        node_degrees = torch.FloatTensor(engineer.degrees)
        
        # 构造此子图的 y
        y_np = np.zeros(N_sub, dtype=np.float32)
        source_indices_sub = [node2idx[s] for s in valid_sources]
        y_np[source_indices_sub] = 1.0
        
        # 3. 训练节点只使用 初始快照(t=0，对应 mat[:, 1]) 和 最终快照(t_idx=最后一个)
        valid_t_indices = list(set([0, n_snapshot_cols - 1]))
        valid_t_indices.sort()
        
        for t_idx in valid_t_indices:
            infected_vec_global = mat[:, t_idx + 1]
            infected_nodes_global = set(np.where(infected_vec_global == 1)[0]) | source_nodes
            
            # 过滤出在此连通快子图内的感染节点
            infected_indices_sub = [node2idx[n] for n in infected_nodes_global if n in node2idx]
            if len(infected_indices_sub) == 0:
                continue
                
            is_final = (t_idx == n_snapshot_cols - 1)
            
            x_observed, k_inf_tensor = engineer._build_observed_features(infected_indices_sub, source_indices_sub)
            
            train_mask = torch.zeros(N_sub, dtype=torch.bool)
            train_mask[infected_indices_sub] = True
            
            data = Data(
                x=torch.FloatTensor(x_observed),
                edge_index=edge_index,
                edge_dist=edge_dist,
                degrees=node_degrees,
                y=torch.FloatTensor(y_np),
                train_mask=train_mask,
                k_inf=k_inf_tensor,
                cascade_id=cascade_idx,
                is_final=is_final,
                global_node_ids=torch.LongTensor(cc_nodes),
            )
            dataset.append(data)
            
        if (cascade_idx + 1) % 100 == 0:
            log_print(logger, f"    已处理 {cascade_idx + 1}/{len(influ)} 个级联...")

    log_print(logger, f"[*] 数据集构建完成, 共 {len(dataset)} 个快照样本")

    # 数据划分
    all_cascade_ids = sorted(set(d.cascade_id for d in dataset))
    random.shuffle(all_cascade_ids)
    n_cas = len(all_cascade_ids)

    train_cas_ids = set(all_cascade_ids[:int(n_cas * TRAIN_RATIO)])
    val_cas_ids   = set(all_cascade_ids[int(n_cas * TRAIN_RATIO):int(n_cas * (TRAIN_RATIO + VAL_RATIO))])
    test_cas_ids  = set(all_cascade_ids[int(n_cas * (TRAIN_RATIO + VAL_RATIO)):])

    train_set = [d for d in dataset if d.cascade_id in train_cas_ids]
    val_set   = [d for d in dataset if d.cascade_id in val_cas_ids and d.is_final]
    test_set  = [d for d in dataset if d.cascade_id in test_cas_ids and d.is_final]
    
    log_print(logger, f"[*] 数据划分: Train={len(train_set)} (只包含首末快照), Val={len(val_set)} (仅最终快照), Test={len(test_set)} (仅最终快照)")
    
    train_loader = DataLoader(train_set, batch_size=1, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=1)
    test_loader = DataLoader(test_set, batch_size=1)
    
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
        dropout=arch_config.get('dropout', 0.3),
        beta=arch_config.get('beta', 1.0),
        lambda_1=arch_config.get('lambda_1', 0.5),
        lambda_2=arch_config.get('lambda_2', 1.0)
    ).to(DEVICE)
    
    param_info = model.get_num_params()
    log_print(logger, "[*] 模型参数:")
    log_print(logger, f"    Encoder: {param_info['encoder']:,}")
    log_print(logger, f"    ClassHead: {param_info['class_head']:,}")
    log_print(logger, f"    Total: {param_info['total']:,}")
    
    criterion = GFRRLoss(
        pos_weight=pos_weight,
        lambda_rank=loss_config.get('lambda_rank', 0.1),
        margin=loss_config.get('margin', 0.15)
    ).to(DEVICE)
    
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY
    )
    
    best_val_f1 = 0
    best_threshold = 0.5
    best_epoch = 0
    save_dir = os.path.join(os.path.dirname(__file__), '..', 'checkpoints_gfrr')
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f'gfrr_lite_subgraph_{data_name}_best.pt')
    
    log_print(logger, f"\n{'='*60}")
    log_print(logger, f"      Epochs: {EPOCHS}")
    log_print(logger, f"      LR: {LR}")
    log_print(logger, f"      Weight Decay: {WEIGHT_DECAY}")
    log_print(logger, f"      Lambda Rank: {loss_config.get('lambda_rank', 0.1)}")
    log_print(logger, f"      pos_weight: {pos_weight:.2f}")
    log_print(logger, f"      hidden_dim: {arch_config.get('hidden_dim', 32)}")
    log_print(logger, f"      数据划分: 训练只用首尾快照，评测/测试仅用最终快照")
    log_print(logger, f"{'='*60}\n")
    
    for epoch in range(EPOCHS):
        avg_loss, loss_comp = train_epoch(model, train_loader, criterion, optimizer, DEVICE)
        
        best_th, _ = find_optimal_threshold_subgraph(model, val_loader, DEVICE)
        val_metrics = evaluate_subgraph(
            model, val_loader, DEVICE,
            threshold=best_th,
            recall_k_values=RECALL_K_VALUES,
            dist_matrix=dist_matrix
        )
        
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
            
        recall_str = " | ".join([f"R@{k}: {val_metrics[f'recall@{k}']:.3f}" for k in RECALL_K_VALUES])
        log_print(logger, f"Epoch {epoch+1:03d} | Loss: {avg_loss:.4f} "
              f"(BCE:{loss_comp['bce']:.3f}, Rank:{loss_comp['rank']:.3f}) | "
              f"Val F1: {val_metrics['f1']:.4f} | {recall_str}")
              
    log_print(logger, f"[*] 最佳 Val F1: {best_val_f1:.4f} @ Epoch {best_epoch}")
    
    checkpoint = torch.load(save_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    test_metrics = evaluate_subgraph(
        model, test_loader, DEVICE,
        threshold=best_threshold,
        recall_k_values=RECALL_K_VALUES,
        dist_matrix=dist_matrix
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
