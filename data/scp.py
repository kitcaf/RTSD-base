"""
Source Count Predictor (SCP) - 增强版
======================================
核心目标：预测传播级联中的源点数量 (k)，为溯源任务提供动态截断阈值。

输入：
    - 静态社交网络图 (Adj)
    - 动态传播级联 (Infected Status)
输出：
    - 预测的源点数量 (Float -> Int)

架构：
    GIN (节点级) -> Pooling -> [拼接图级特征] -> MLP -> 回归输出
    
图级特征 (5个维度):
    1. 连通性 (Connectivity): 感染子图的连通分量数
    2. 形状 (Shape): 感染子图的直径/半径比
    3. 边缘效应 (Boundary): 边界节点占比
    4. 紧密度 (Density): 感染子图的边密度
    5. 中心性分布 (Centrality): 度中心性的方差 (多中心 vs 单中心)
"""

import numpy as np
import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GINConv, global_add_pool, global_mean_pool, global_max_pool
import pickle
import os
import random
import math

# ==========================================
# 0. 随机种子设置
# ==========================================
def setup_seed(seed=42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# 数据集选择
datasetsNames = ['android', 'christianity', 'douban', 'twitter']
DATASET_IDX = 0  # 修改这里选择数据集
dataName = datasetsNames[DATASET_IDX]
fileName = f"{dataName}_25c.SG"

# ==========================================
# 1. 数据加载模块
# ==========================================
class DataContainer:
    pass

class RobustUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module.startswith('data'):
            return DataContainer
        try:
            return super().find_class(module, name)
        except ModuleNotFoundError:
            return DataContainer

def load_raw_data(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"未找到文件: {file_path}。请确保文件在当前目录下。")
    
    print(f"[*] 正在加载数据: {file_path} ...")
    with open(file_path, 'rb') as f:
        try:
            data = pickle.load(f)
        except (ModuleNotFoundError, AttributeError):
            f.seek(0)
            data = RobustUnpickler(f).load()

    adj = None
    if hasattr(data, 'adj_matrix'): adj = data.adj_matrix
    elif isinstance(data, dict) and 'adj_matrix' in data: adj = data['adj_matrix']
    elif isinstance(data, (list, tuple)): adj = data[0]
    
    influ = None
    if hasattr(data, 'influ_mat_list'): influ = data.influ_mat_list
    elif isinstance(data, dict) and 'influ_mat_list' in data: influ = data['influ_mat_list']
    elif isinstance(data, (list, tuple)): influ = data[1]

    if hasattr(adj, 'toarray'): adj = adj.toarray()
    if isinstance(influ, list): influ = np.array(influ)
    
    return adj, influ

# ==========================================
# 2. 特征工程 (节点级 + 图级)
# ==========================================
class CountFeatureEngineer:
    """
    特征工程模块，提取两类特征：
    
    === 节点级特征 (Node-Level, 输入GIN) ===
    1. is_infected: 节点是否感染
    2. norm_degree: 归一化节点度数
    3. norm_neighbor_inf: 归一化感染邻居数
    
    === 图级特征 (Graph-Level, GIN聚合后拼接) ===
    1. 感染规模 (log_infected): log(感染节点数) - 最重要的特征!
    2. 感染率 (infection_rate): 感染节点占总节点比例
    3. 连通分量数 (num_components): 感染子图的连通分量数 (关键特征)
    4. 边缘效应 (boundary_ratio): 边界节点占感染节点比例
    5. 紧密度 (density): 感染子图的边密度
    6. 中心性分布 (centrality_var): 度中心性的变异系数
    7. 平均聚类系数 (avg_clustering): 感染子图的平均聚类系数
    8. 形状 (shape_ratio): 直径/半径比 (反映扩散形态)
    """
    
    GRAPH_FEATURE_DIM = 8  # 图级特征维度 (增加到8维)
    
    def __init__(self, adj_matrix):
        self.adj_matrix = adj_matrix
        self.G = nx.from_numpy_array(adj_matrix)
        self.num_nodes = adj_matrix.shape[0]
        
        # 预计算静态特征
        self.degrees = np.array([d for n, d in self.G.degree()])
        self.max_degree = self.degrees.max() if self.degrees.max() > 0 else 1
        self.neighbors_list = [list(self.G.neighbors(i)) for i in range(self.num_nodes)]

    def _compute_graph_features(self, infected_indices, infected_set):
        """
        计算感染子图的8个图级特征
        返回: np.array of shape (8,)
        """
        num_infected = len(infected_indices)
        
        if num_infected == 0:
            return np.zeros(8, dtype=np.float32)
        
        # 构建感染子图
        infected_subgraph = self.G.subgraph(infected_indices).copy()
        
        # ====== 1. 感染规模 (最重要的特征!) ======
        # 感染节点数与源点数往往正相关
        # 使用log变换压缩范围
        log_infected = np.log(num_infected + 1) / np.log(self.num_nodes + 1)
        
        # ====== 2. 感染率 ======
        infection_rate = num_infected / self.num_nodes
        
        # ====== 3. 连通分量数 (关键特征) ======
        # 多源点往往导致多个独立的感染簇
        num_components = nx.number_connected_components(infected_subgraph)
        # 直接使用分量数的log变换，而不是归一化
        log_components = np.log(num_components + 1) / np.log(num_infected + 1)
        
        # ====== 4. 边缘效应: 边界节点比例 ======
        boundary_count = 0
        for node in infected_indices:
            has_uninfected_neighbor = any(nb not in infected_set for nb in self.neighbors_list[node])
            if has_uninfected_neighbor:
                boundary_count += 1
        boundary_ratio = boundary_count / num_infected
        
        # ====== 5. 紧密度: 子图边密度 ======
        num_edges = infected_subgraph.number_of_edges()
        max_possible_edges = num_infected * (num_infected - 1) / 2
        density = num_edges / max_possible_edges if max_possible_edges > 0 else 0.0
        
        # ====== 6. 中心性分布: 度中心性的变异系数 ======
        if num_infected > 1:
            subgraph_degrees = [infected_subgraph.degree(n) for n in infected_indices]
            mean_deg = np.mean(subgraph_degrees)
            std_deg = np.std(subgraph_degrees)
            centrality_cv = std_deg / max(mean_deg, 1)
            centrality_var = min(centrality_cv / 2.0, 1.0)
        else:
            centrality_var = 0.0
        
        # ====== 7. 平均聚类系数 ======
        try:
            avg_clustering = nx.average_clustering(infected_subgraph)
        except:
            avg_clustering = 0.0
        
        # ====== 8. 形状: 直径/半径比 ======
        try:
            if nx.is_connected(infected_subgraph) and num_infected > 1:
                diameter = nx.diameter(infected_subgraph)
                radius = nx.radius(infected_subgraph)
                shape_ratio = diameter / max(radius, 1)
                shape_ratio = min(shape_ratio / 4.0, 1.0)
            else:
                largest_cc = max(nx.connected_components(infected_subgraph), key=len)
                if len(largest_cc) > 1:
                    largest_subgraph = infected_subgraph.subgraph(largest_cc)
                    diameter = nx.diameter(largest_subgraph)
                    radius = nx.radius(largest_subgraph)
                    shape_ratio = min((diameter / max(radius, 1)) / 4.0, 1.0)
                else:
                    shape_ratio = 0.0
        except:
            shape_ratio = 0.0
        
        return np.array([log_infected, infection_rate, log_components, boundary_ratio, 
                        density, centrality_var, avg_clustering, shape_ratio], dtype=np.float32)

    def generate_dataset(self, influ_list):
        print("[*] 正在构建特征...")
        print(f"    节点级特征: 3维 (is_infected, degree, neighbor_inf)")
        print(f"    图级特征: 8维 (感染规模, 感染率, 连通分量, 边缘效应, 紧密度, 中心性, 聚类系数, 形状)")
        dataset = []
        
        # 构建边索引 (静态图共享)
        rows, cols = np.where(self.adj_matrix > 0)
        edge_index = torch.LongTensor(np.array([rows, cols]))
        
        log_max_degree = np.log(self.max_degree + 1)

        for cascade_idx, mat in enumerate(influ_list):
            source_vec = mat[:, 0]
            infected_vec = mat[:, 1]
            
            # 标签: 源点数量
            num_sources = np.sum(source_vec)
            
            infected_indices = np.where(infected_vec == 1)[0]
            infected_set = set(infected_indices)
            
            if len(infected_indices) == 0:
                continue
            
            # ========================================
            # 计算图级特征 (5维)
            # ========================================
            graph_features = self._compute_graph_features(infected_indices, infected_set)
            
            # ========================================
            # 计算节点级特征 (3维)
            # ========================================
            k_inf_all = np.zeros(self.num_nodes)
            for i in range(self.num_nodes):
                k_inf_all[i] = sum(1 for nb in self.neighbors_list[i] if nb in infected_set)

            x_np = np.zeros((self.num_nodes, 3), dtype=np.float32)
            
            for i in range(self.num_nodes):
                f1 = 1.0 if i in infected_set else 0.0
                f2 = np.log(self.degrees[i] + 1) / log_max_degree
                f3 = k_inf_all[i] / self.max_degree
                x_np[i] = [f1, f2, f3]

            # 创建 Data 对象, 包含图级特征
            data = Data(
                x=torch.FloatTensor(x_np),
                edge_index=edge_index,
                y=torch.FloatTensor([num_sources]),
                graph_feat=torch.FloatTensor(graph_features)  # 图级特征存储在这里
            )
            dataset.append(data)
            
            if (cascade_idx + 1) % 100 == 0:
                print(f"    已处理 {cascade_idx + 1}/{len(influ_list)} 个级联...")
            
        print(f"[*] 数据集构建完成. 样本数: {len(dataset)}")
        return dataset

# ==========================================
# 3. 模型: GIN + 图级特征融合 (简化版，适合小数据集)
# ==========================================
class SourceCountPredictor(torch.nn.Module):
    """
    简化架构 (适合小数据集):
        节点特征 -> GIN(1层) -> Pooling(sum/mean) -> [拼接图级特征] -> MLP -> 输出
        
    输入维度:
        - 节点特征: 3维
        - 图级特征: 8维 (在Pooling后拼接)
    
    简化设计:
        - 减少GIN层数 (2->1)
        - 减少pooling类型 (3->2)
        - 更简单的MLP
    """
    def __init__(self, num_node_features=3, num_graph_features=8, hidden_dim=32):
        super(SourceCountPredictor, self).__init__()
        
        self.num_graph_features = num_graph_features
        
        # --- GIN Layer (仅1层，减少过拟合) ---
        self.mlp1 = nn.Sequential(
            nn.Linear(num_node_features, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.conv1 = GINConv(self.mlp1)

        # --- Regression Head ---
        # 输入维度 = hidden_dim * 2 (sum + mean pooling) + num_graph_features
        regressor_input_dim = hidden_dim * 2 + num_graph_features
        
        self.regressor = nn.Sequential(
            nn.Linear(regressor_input_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 1)
        )

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        graph_feat = data.graph_feat  # DataLoader concatenates into 1D, need reshape
        
        # 获取batch中的图数量
        num_graphs = batch.max().item() + 1
        # 重塑 graph_feat: [num_graphs * num_graph_features] -> [num_graphs, num_graph_features]
        graph_feat = graph_feat.view(num_graphs, self.num_graph_features)
        
        # Node Embeddings via GIN (仅1层)
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        
        # Readout (简化为2种pooling)
        x_sum = global_add_pool(x, batch)   # 捕捉规模
        x_mean = global_mean_pool(x, batch) # 捕捉平均强度
        
        # 拼接 GIN 输出和图级特征
        gin_embed = torch.cat([x_sum, x_mean], dim=1)  # [B, hidden*2]
        combined = torch.cat([gin_embed, graph_feat], dim=1)  # [B, hidden*2 + 8]
        
        # Prediction
        out = self.regressor(combined).squeeze()
        return out


# ==========================================
# 3.5 备选: 纯MLP模型 (更适合小数据集)
# ==========================================
class SourceCountMLP(torch.nn.Module):
    """
    纯MLP模型，仅使用图级特征
    对于小数据集，简单模型往往效果更好
    """
    def __init__(self, num_graph_features=8, hidden_dim=64):
        super(SourceCountMLP, self).__init__()
        
        self.num_graph_features = num_graph_features
        
        self.mlp = nn.Sequential(
            nn.Linear(num_graph_features, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, data):
        graph_feat = data.graph_feat
        
        # 处理batch
        if hasattr(data, 'batch') and data.batch is not None:
            num_graphs = data.batch.max().item() + 1
            graph_feat = graph_feat.view(num_graphs, self.num_graph_features)
        else:
            graph_feat = graph_feat.view(-1, self.num_graph_features)
        
        out = self.mlp(graph_feat).squeeze()
        return out

# ==========================================
# 4. 损失函数: Asymmetric Loss (非对称损失)
# ==========================================
class AsymmetricMSELoss(nn.Module):
    def __init__(self, alpha=3.0):
        super(AsymmetricMSELoss, self).__init__()
        self.alpha = alpha
        
    def forward(self, pred, target):
        diff = pred - target
        loss = diff ** 2
        weighted_loss = torch.where(diff < 0, self.alpha * loss, loss)
        return weighted_loss.mean()

# ==========================================
# 5. 训练与评估流程
# ==========================================
def evaluate(model, loader, device):
    model.eval()
    mae_list = []
    mse_list = []
    raw_preds = []
    raw_targets = []
    
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            out = model(data)
            
            if out.dim() == 0: out = out.unsqueeze(0)
            
            preds = out
            targets = data.y
            
            mae = torch.abs(preds - targets).mean()
            mse = ((preds - targets) ** 2).mean()
            
            mae_list.append(mae.item())
            mse_list.append(mse.item())
            
            raw_preds.extend(preds.cpu().tolist())
            raw_targets.extend(targets.cpu().tolist())
            
    avg_mae = np.mean(mae_list)
    avg_mse = np.mean(mse_list)
    
    # 转换为numpy数组
    preds_float = np.array(raw_preds)
    all_targets = np.array(raw_targets)
    
    # 向上取整 (ceil) - 因为4.1个源点意味着要截取Top-5
    preds_int = np.ceil(preds_float)
    
    # --- 指标 A: 样本级覆盖率 (Sample-level Coverage) ---
    # 只有 "预测值 >= 真实值" 才算成功
    sample_success_mask = (preds_int >= all_targets)
    sample_cov_rate = np.mean(sample_success_mask)
    
    # --- 指标 B: 源点级覆盖率 (Source-level Coverage) ---
    # 计算每个样本实际能覆盖多少个: min(预测, 真实)
    actual_covered = np.minimum(preds_int, all_targets)
    # 计算每个样本的比例: 覆盖数 / 真实数
    ratios = actual_covered / (all_targets + 1e-9)
    # 对所有样本取平均
    source_cov_rate = np.mean(ratios)
    
    # 低估率 (使用原始浮点预测值)
    under_count = np.sum(preds_float < all_targets)
    total_samples = len(all_targets)
    under_rate = under_count / total_samples if total_samples > 0 else 0.0
    
    return avg_mae, avg_mse, under_rate, sample_cov_rate, source_cov_rate, raw_preds, raw_targets

def main():
    # ===== 配置 =====
    SEED = 2025
    setup_seed(SEED)
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[*] 使用设备: {DEVICE}")
    
    # 超参数
    EPOCHS = 100
    LR = 0.005
    BATCH_SIZE = 16  # 小数据集用小batch
    ALPHA = 5.0      # 增大alpha，减少低估
    USE_MLP = True   # True: 使用纯MLP (推荐小数据集), False: 使用GIN
    
    # ===== 加载数据 =====
    try:
        adj, influ = load_raw_data(fileName)
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    # ===== 特征工程 =====
    engineer = CountFeatureEngineer(adj)
    dataset = engineer.generate_dataset(influ)
    
    # ===== 分析标签分布 =====
    labels = [d.y.item() for d in dataset]
    print(f"\n[*] 标签分布分析:")
    print(f"    最小值: {min(labels):.0f}, 最大值: {max(labels):.0f}")
    print(f"    均值: {np.mean(labels):.2f}, 中位数: {np.median(labels):.2f}")
    print(f"    标准差: {np.std(labels):.2f}")
    
    # ===== 数据划分 =====
    random.shuffle(dataset)
    n = len(dataset)
    train_set = dataset[:int(n*0.7)]
    val_set = dataset[int(n*0.7):int(n*0.85)]
    test_set = dataset[int(n*0.85):]
    
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE)
    
    # ===== 模型初始化 =====
    if USE_MLP:
        print("纯MLP")
        model = SourceCountMLP(
            num_graph_features=CountFeatureEngineer.GRAPH_FEATURE_DIM
        ).to(DEVICE)
        model_name = "纯MLP (图级特征)"
    else:
        model = SourceCountPredictor(
            num_node_features=3, 
            num_graph_features=CountFeatureEngineer.GRAPH_FEATURE_DIM
        ).to(DEVICE)
        model_name = "GIN + 图级特征融合"
    
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=20)
    criterion = AsymmetricMSELoss(alpha=ALPHA).to(DEVICE)
    
    print(f"\n{'='*60}")
    print(f"[*] 任务: 源点数量预测 (Source Count Estimation)")
    print(f"[*] 模型: {model_name}")
    print(f"[*] 图级特征: 感染规模, 感染率, 连通分量, 边缘, 密度, 中心性, 聚类, 形状")
    print(f"[*] Loss: Asymmetric MSE (Alpha={ALPHA})")
    print(f"[*] 样本数: Train={len(train_set)}, Val={len(val_set)}, Test={len(test_set)}")
    print(f"{'='*60}\n")
    
    # ===== 训练循环 =====
    best_val_mae = float('inf')
    
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        
        for data in train_loader:
            data = data.to(DEVICE)
            optimizer.zero_grad()
            
            out = model(data)
            
            if out.dim() == 0: out = out.unsqueeze(0)
            
            loss = criterion(out, data.y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        avg_loss = total_loss / len(train_loader)
        
        # 验证
        val_mae, val_mse, val_ur, val_sample_cr, val_source_cr, _, _ = evaluate(model, val_loader, DEVICE)
        
        # 学习率调度
        scheduler.step(val_mae)
        
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            torch.save(model.state_dict(), f'best_count_model_{dataName}.pth')
            
        if (epoch + 1) % 10 == 0:
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch+1:03d} | Loss: {avg_loss:.4f} | Val MAE: {val_mae:.4f} | Sample CR: {val_sample_cr:.2%} | Source CR: {val_source_cr:.2%} | LR: {current_lr:.6f}")
            
    print(f"\n[*] 训练结束. 最佳 Val MAE: {best_val_mae:.4f}")
    
    # ===== 测试 =====
    model.load_state_dict(torch.load(f'best_count_model_{dataName}.pth'))
    test_mae, test_mse, test_ur, test_sample_cr, test_source_cr, preds, targets = evaluate(model, test_loader, DEVICE)
    
    print(f"\n{'='*60}")
    print("最终测试结果 (Final Test Metrics)")
    print(f"{'='*60}")
    print(f"MAE (平均绝对误差): {test_mae:.4f}")
    print(f"MSE (均方误差)    : {test_mse:.4f}")
    print(f"UR  (低估率)      : {test_ur:.2%}  (越低越好)")
    print(f"样本级覆盖率 (Sample CR): {test_sample_cr:.2%}  (ceil后 pred>=true 的比例)")
    print(f"源点级覆盖率 (Source CR): {test_source_cr:.2%}  (平均能覆盖多少比例的真实源点)")
    
    # 展示部分预测结果
    print("\n预测示例 (Pred vs True):")
    for i in range(10):
        if i < len(preds):
            p = preds[i]
            t = targets[i]
            print(f"  样本 {i}: 预测 {p:.2f} (取整 {math.ceil(p)}) vs 真实 {t:.0f}")

if __name__ == "__main__":
    main()