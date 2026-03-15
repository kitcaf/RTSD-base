"""
Feature Selection & Importance Analysis using XGBoost
===================================================
目标：
1. 量化 V5 中 11 个特征的真实贡献度。
2. 识别冗余特征（共线性）。
3. 自动推荐最优特征子集 (Top-N)。
"""

import numpy as np
import networkx as nx
import xgboost as xgb
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import RFE
from sklearn.metrics import f1_score, precision_score, recall_score
import pickle
import os
import warnings

# 复用你代码中的类（为了保持一致性，直接复制部分核心逻辑）
# ==========================================
class DataContainer: pass
class RobustUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module.startswith('data'): return DataContainer
        return super().find_class(module, name)

def load_raw_data(file_path):
    if not os.path.exists(file_path): raise FileNotFoundError(f"Missing: {file_path}")
    print(f"[*] Loading {file_path}...")
    with open(file_path, 'rb') as f:
        try: data = pickle.load(f)
        except: f.seek(0); data = RobustUnpickler(f).load()
    adj = getattr(data, 'adj_matrix', None)
    if adj is None and isinstance(data, (list, tuple)): adj = data[0]
    influ = getattr(data, 'influ_mat_list', None)
    if influ is None and isinstance(data, (list, tuple)): influ = data[1]
    if hasattr(adj, 'toarray'): adj = adj.toarray()
    if isinstance(influ, list): influ = np.array(influ)
    return adj, influ

# ==========================================
# 2. 特征工程 (完全复用 V5，保证特征一致)
# ==========================================
class FeatureEngineerV5_Analysis:
    def __init__(self, adj_matrix):
        self.adj_matrix = adj_matrix
        self.G = nx.from_numpy_array(adj_matrix)
        self.num_nodes = adj_matrix.shape[0]
        self.degrees = np.array([d for n, d in self.G.degree()])
        self.max_degree = self.degrees.max() if self.degrees.max() > 0 else 1
        self.neighbors_list = [list(self.G.neighbors(i)) for i in range(self.num_nodes)]
        self.global_rank = np.zeros(self.num_nodes)
        sorted_indices = np.argsort(-self.degrees)
        for rank, idx in enumerate(sorted_indices): self.global_rank[idx] = rank

    def compute_subgraph_kcore(self, infected_indices):
        if len(infected_indices) < 2: return {i: 0 for i in infected_indices}
        G_sub = self.G.subgraph(infected_indices)
        try: return nx.core_number(G_sub)
        except: return {i: 0 for i in infected_indices}

    def generate_dataset(self, influ_list):
        print("[*] Generating Features (V5 Logic)...")
        # 我们需要把所有级联的数据打平，变成一个大的 X, y 矩阵
        all_X = []
        all_y = []
        
        ALPHA = 2.0 
        log_max_degree = np.log(self.max_degree + 1)

        for cascade_idx, mat in enumerate(influ_list):
            source_vec = mat[:, 0]
            infected_vec = mat[:, 1]
            true_src = np.where(source_vec == 1)[0]
            infected = np.where(infected_vec == 1)[0]
            if len(true_src) == 0 or len(infected) == 0: continue
            
            infected_set = set(infected)
            kcore_sub = self.compute_subgraph_kcore(infected)
            max_sub_kcore = max(kcore_sub.values()) if kcore_sub else 1
            if max_sub_kcore == 0: max_sub_kcore = 1
            
            k_inf_all = np.zeros(self.num_nodes)
            # 优化：只计算感染节点及其邻居的k_inf，不需要全图算
            # 但为了逻辑一致，我们只关注 infected_indices 里的节点
            for node in infected:
                k_inf_all[node] = sum(1 for nb in self.neighbors_list[node] if nb in infected_set)
                
            sub_rank = {n: 0 for n in infected}
            if len(infected) > 0:
                k_inf_infected = k_inf_all[infected]
                # 局部排序
                local_indices = np.argsort(-k_inf_infected)
                for r, idx in enumerate(local_indices):
                    sub_rank[infected[idx]] = r
            
            max_kinf = k_inf_all.max() if k_inf_all.max() > 0 else 1

            # 只收集感染节点的样本 (Imbalanced Classification)
            for i in infected:
                label = 1 if i in true_src else 0
                
                # Feat 1: is_infected (Always 1 in this subset, but keep for index consistency)
                feat_1 = 1.0 
                # Feat 2: norm_deg
                feat_2 = np.log(self.degrees[i] + 1) / log_max_degree
                # Feat 3: norm_inf_count
                feat_3 = k_inf_all[i] / self.max_degree
                # Feat 4: log_deg
                feat_4 = np.log(self.degrees[i] + 1e-5) / 5.0
                # Feat 5: i_score
                feat_5 = (k_inf_all[i] + ALPHA * np.log(self.degrees[i] + 1e-5)) / 50.0
                # Feat 6: kcore_sub
                feat_6 = kcore_sub.get(i, 0) / max_sub_kcore
                # Feat 7: max_neighbor_kinf
                nbs = self.neighbors_list[i]
                inf_nbs = [nb for nb in nbs if nb in infected_set]
                max_nb = max([k_inf_all[nb] for nb in inf_nbs]) if inf_nbs else 0
                feat_7 = np.log(max_nb + 1) / np.log(max_kinf + 1) if max_kinf > 0 else 0
                # Feat 8: mismatch
                feat_8 = (sub_rank[i] - self.global_rank[i]) / self.num_nodes
                # Feat 9: kinf_rank_nb
                stronger = sum(1 for nb in nbs if k_inf_all[nb] > k_inf_all[i])
                feat_9 = stronger / (len(nbs) + 1e-6) if nbs else 0
                # Feat 10: is_peak
                max_nb_val = max([k_inf_all[nb] for nb in nbs]) if nbs else 0
                feat_10 = 1.0 if k_inf_all[i] >= max_nb_val else 0.0
                # Feat 11: bridge_score
                sum_nb_kinf = sum(k_inf_all[nb] for nb in inf_nbs) if inf_nbs else 0
                bridge = (k_inf_all[i] * self.degrees[i]) / (sum_nb_kinf + 1e-6)
                feat_11 = np.log(bridge + 1) / 10.0

                all_X.append([feat_1, feat_2, feat_3, feat_4, feat_5, feat_6, 
                              feat_7, feat_8, feat_9, feat_10, feat_11])
                all_y.append(label)
                
            if (cascade_idx+1) % 100 == 0: print(f"  Processed {cascade_idx+1} cascades...")
            
        return np.array(all_X), np.array(all_y)

# ==========================================
# 3. 分析主程序
# ==========================================
def main():
    # 数据集配置
    DATASET_IDX = 0 # 0:Android, 1:Christianity, 2:Douban, 3:Twitter
    datasets = ['android_25c.SG', 'christianity_25c.SG', 'douban_25c.SG', 'twitter_25c.SG']
    data_path = datasets[DATASET_IDX]
    
    # 1. 加载与生成
    try:
        adj, influ = load_raw_data(data_path)
        engineer = FeatureEngineerV5_Analysis(adj)
        X, y = engineer.generate_dataset(influ)
    except Exception as e:
        print(f"Error: {e}")
        return

    print(f"\n[*] Data Shape: X={X.shape}, y={y.shape}, Pos Ratio={y.mean():.4f}")
    
    feature_names = [
        "1. is_infected", "2. norm_deg", "3. norm_inf_count", "4. log_deg",
        "5. i_score", "6. kcore_sub", "7. max_nb_kinf", "8. mismatch",
        "9. kinf_rank_nb", "10. is_peak", "11. bridge"
    ]
    
    # 2. 划分数据集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2025, stratify=y)
    
    # 3. XGBoost 训练
    print("\n[*] Training XGBoost Classifier...")
    # scale_pos_weight 处理正负样本不平衡 (Pos很少)
    scale_weight = (len(y) - sum(y)) / sum(y)
    model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_weight,
        random_state=2025,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    
    # 简单评估
    y_pred = model.predict(X_test)
    print(f"Test F1: {f1_score(y_test, y_pred):.4f}")
    print(f"Test Pre: {precision_score(y_test, y_pred):.4f}")
    print(f"Test Rec: {recall_score(y_test, y_pred):.4f}")
    
    # 4. 特征重要性分析 (Gain)
    importance = model.feature_importances_
    df_imp = pd.DataFrame({'Feature': feature_names, 'Gain': importance})
    df_imp = df_imp.sort_values(by='Gain', ascending=False)
    
    print("\n[Feature Importance Ranking (Gain)]")
    print(df_imp)
    
    # 画图
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Gain', y='Feature', data=df_imp, palette='viridis')
    plt.title(f'XGBoost Feature Importance - {datasets[DATASET_IDX]}')
    plt.tight_layout()
    plt.show() # 如果在 notebook 中
    plt.savefig('feature_importance.png')
    print("Saved feature_importance.png")
    
    # 5. 共线性检查 (Redundancy Check)
    print("\n[Collinearity Check (Correlation > 0.95)]")
    df_X = pd.DataFrame(X, columns=feature_names)
    corr_matrix = df_X.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    
    high_corr = [column for column in upper.columns if any(upper[column] > 0.95)]
    if high_corr:
        print(f"Highly Correlated Features (Candidates to Drop): {high_corr}")
        # 打印具体的对
        for col in high_corr:
            print(f"  - {col} is correlated with: {upper.index[upper[col] > 0.95].tolist()}")
    else:
        print("No extreme collinearity found.")
        
    # 6. RFE 自动筛选 (Recursive Feature Elimination)
    print("\n[*] Running RFE to find Top-5 features...")
    rfe = RFE(estimator=model, n_features_to_select=5, step=1)
    rfe.fit(X_train, y_train)
    
    print("Top 5 Selected Features:")
    for i, selected in enumerate(rfe.support_):
        if selected:
            print(f"  {feature_names[i]}")

if __name__ == "__main__":
    main()