"""
GFRR 特征工程模块
生成观测态特征 (x_observed) 和 源点态特征 (x_source)

设计原则:
    - x_observed: 基于完整感染快照计算的 14 维特征
    - x_source: 基于源点初始状态的 14 维特征
        - 静态特征保持不变: is_source, norm_deg, log_deg
        - 传播相关特征设为初始值 (0 或合理默认值)
"""
import numpy as np
import scipy.sparse as sp
import networkx as nx
import torch
import os
from torch_geometric.data import Data
from scipy.stats import rankdata


class FeatureEngineerGFRR:
    """
    GFRR 特征工程器
    
    生成两套特征:
        1. x_observed [N, 14]: 观测态特征 (基于感染快照)
        2. x_source [N, 14]: 源点态特征 (初始状态)
    
    特征维度说明:
        [0]  state_flag: 状态标志 (observed: is_infected, source: is_source)
        [1]  norm_deg: 归一化全局度 (静态, 共享)
        [2]  norm_inf_count: 归一化感染邻居数
        [3]  log_deg: 对数度数 (静态, 共享)
        [4]  i_score: 组合得分
        [5]  kcore_sub: 子图K-Core数
        [6]  max_neighbor_kinf: 感染邻居最大k_inf
        [7]  mismatch_ratio: 拓扑失配比
        [8]  kinf_rank_in_neighborhood: 邻域内k_inf排名
        [9]  is_local_kinf_peak: 是否局部k_inf峰值
        [10] bridge_score: 桥梁得分
        [11] subgraph_closeness: 子图接近中心性
        [12] subgraph_eccentricity: 子图偏心率
        [13] outward_ratio: 传播方向比
    """
    
    # 静态特征索引 (不依赖传播状态)
    STATIC_FEATURE_INDICES = [1, 3]  # norm_deg, log_deg
    
    # 特征名称映射
    FEATURE_NAMES = [
        'state_flag', 'norm_deg', 'norm_inf_count', 'log_deg',
        'i_score', 'kcore_sub', 'max_neighbor_kinf', 'mismatch_ratio',
        'kinf_rank_in_neighborhood', 'is_local_kinf_peak', 'bridge_score',
        'subgraph_closeness', 'subgraph_eccentricity', 'outward_ratio'
    ]
    
    def __init__(self, adj_matrix, ablation_feature_idx=None):
        """
        初始化特征工程器
        
        Args:
            adj_matrix: 邻接矩阵 [N, N]
            ablation_feature_idx: 要消融的特征索引 (0-13), None表示不消融
        """
        self.adj_matrix = adj_matrix
        self.G = nx.from_numpy_array(adj_matrix)
        self.num_nodes = adj_matrix.shape[0]
        self.ablation_feature_idx = ablation_feature_idx
        
        # 预计算全图静态特征
        self.degrees = np.array([d for n, d in self.G.degree()])
        self.max_degree = max(self.degrees.max(), 1)
        self.neighbors_list = [list(self.G.neighbors(i)) for i in range(self.num_nodes)]
        
        # 全局度数排名
        self.global_rank = np.zeros(self.num_nodes)
        sorted_indices = np.argsort(-self.degrees)
        for rank, idx in enumerate(sorted_indices):
            self.global_rank[idx] = rank
        
        # 度数百分位
        degree_ranks = rankdata(self.degrees, method='average')
        self.degree_percentile = (degree_ranks - 1) / max(self.num_nodes - 1, 1)
        
        # 预计算全局桥梁得分 (仅基于度数)
        self._precompute_global_bridge_scores()
    
    def _precompute_global_bridge_scores(self):
        """预计算基于全局拓扑的桥梁得分"""
        self.global_bridge_scores = np.zeros(self.num_nodes)
        for i in range(self.num_nodes):
            my_degree = self.degrees[i]
            nbs = self.neighbors_list[i]
            if len(nbs) > 0:
                sum_nb_degree = sum(self.degrees[nb] for nb in nbs)
                bridge = (my_degree * my_degree) / (sum_nb_degree + 1e-6)
                self.global_bridge_scores[i] = np.log(bridge + 1) / 10.0
    
    def _compute_subgraph_kcore(self, node_indices):
        """计算子图的 K-Core 数"""
        if len(node_indices) < 2:
            return {i: 0 for i in node_indices}
        
        G_sub = self.G.subgraph(node_indices).copy()
        if G_sub.number_of_edges() == 0:
            return {i: 0 for i in node_indices}
        
        try:
            return nx.core_number(G_sub)
        except:
            return {i: 0 for i in node_indices}
    
    def _compute_subgraph_centralities(self, node_indices):
        """计算子图的中心性特征"""
        closeness = {i: 0.0 for i in node_indices}
        eccentricity = {i: 0.0 for i in node_indices}
        max_ecc = 1.0
        
        if len(node_indices) < 2:
            return closeness, eccentricity, max_ecc
        
        G_sub = self.G.subgraph(node_indices).copy()
        if G_sub.number_of_edges() == 0:
            return closeness, eccentricity, max_ecc
        
        # 处理连通性
        if nx.is_connected(G_sub):
            target_graph = G_sub
        else:
            largest_cc = max(nx.connected_components(G_sub), key=len)
            target_graph = G_sub.subgraph(largest_cc).copy()
        
        if target_graph.number_of_nodes() >= 2:
            try:
                closeness.update(nx.closeness_centrality(target_graph))
            except:
                pass
            try:
                ecc = nx.eccentricity(target_graph)
                eccentricity.update(ecc)
                max_ecc = max(ecc.values()) if ecc else 1.0
            except:
                pass
        
        return closeness, eccentricity, max(max_ecc, 1.0)
    
    def _apply_ablation(self, features):
        """
        应用特征消融：移除指定特征列
        
        Args:
            features: [N, 14] 特征矩阵
        
        Returns:
            ablated_features: [N, 13] 消融后的特征矩阵
        """
        if self.ablation_feature_idx is None:
            return features
        
        # 删除指定列
        ablated = np.delete(features, self.ablation_feature_idx, axis=1)
        return ablated
    
    def get_num_features(self):
        """返回当前特征维度"""
        return 13 if self.ablation_feature_idx is not None else 14
    
    def _extract_cc_info(self, infected_indices):
        """
        提取连通分量信息
        
        Args:
            infected_indices: 感染节点索引
        
        Returns:
            cc_labels: [N] CC标签 (-1表示未感染节点)
            max_cc_id: 最大CC的ID
            cc_sizes: 每个CC的大小
        """
        from scipy.sparse.csgraph import connected_components
        
        # 构建感染子图
        subgraph = self.adj_matrix[np.ix_(infected_indices, infected_indices)]
        n_cc, cc_labels_local = connected_components(subgraph, directed=False)
        
        # 映射到全图节点
        cc_labels = np.full(self.num_nodes, -1, dtype=int)
        cc_labels[infected_indices] = cc_labels_local
        
        # 统计每个CC大小
        cc_sizes = np.bincount(cc_labels_local)
        max_cc_id = np.argmax(cc_sizes)
        
        return cc_labels, max_cc_id, cc_sizes
    
    def _build_observed_features(self, infected_indices, source_indices):
        """
        构建观测态特征 (基于感染快照)
        
        Args:
            infected_indices: 感染节点索引
            source_indices: 源点索引
        
        Returns:
            x_observed: [N, 14] 观测态特征
            k_inf_tensor: [N] 感染邻居数
            cc_labels: [N] CC标签
            max_cc_id: 最大CC的ID
        """
        infected_set = set(infected_indices)
        
        # 提取CC信息
        cc_labels, max_cc_id, cc_sizes = self._extract_cc_info(infected_indices)
        ALPHA = 2.0
        
        # 计算子图特征
        kcore_sub = self._compute_subgraph_kcore(infected_indices)
        max_kcore = max(kcore_sub.values(), default=1) or 1
        
        closeness, eccentricity, max_ecc = self._compute_subgraph_centralities(infected_indices)
        
        # 计算 k_inf
        k_inf_all = np.zeros(self.num_nodes)
        for i in range(self.num_nodes):
            k_inf_all[i] = sum(1 for nb in self.neighbors_list[i] if nb in infected_set)
        max_kinf = max(k_inf_all.max(), 1)
        
        # 子图内 k_inf 排名
        sub_rank = np.zeros(self.num_nodes)
        k_inf_infected = k_inf_all[infected_indices]
        sorted_idx = np.argsort(-k_inf_infected)
        for rank, idx in enumerate(sorted_idx):
            sub_rank[infected_indices[idx]] = rank
        
        # 构建特征矩阵
        x_observed = np.zeros((self.num_nodes, 14), dtype=np.float32)
        
        for i in range(self.num_nodes):
            is_infected = 1.0 if i in infected_set else 0.0
            norm_deg = self.degree_percentile[i]
            log_deg = np.log(self.degrees[i] + 1e-5) / 5.0
            
            inf_count = k_inf_all[i]
            norm_inf_count = inf_count / self.max_degree
            
            raw_i_score = inf_count + ALPHA * np.log(self.degrees[i] + 1e-5)
            norm_i_score = raw_i_score / 50.0
            
            kcore_val = kcore_sub.get(i, 0) if i in infected_set else 0
            norm_kcore = kcore_val / max_kcore
            
            # 感染邻居相关
            nbs = self.neighbors_list[i]
            infected_nbs = [nb for nb in nbs if nb in infected_set]
            
            max_nb_kinf = max((k_inf_all[nb] for nb in infected_nbs), default=0)
            norm_max_nb_kinf = np.log(max_nb_kinf + 1) / np.log(max_kinf + 1) if max_kinf > 0 else 0
            
            mismatch = (sub_rank[i] - self.global_rank[i]) / self.num_nodes if i in infected_set else 0.0
            
            # 邻域 k_inf 排名
            if len(nbs) > 0:
                my_kinf = k_inf_all[i]
                stronger_count = sum(1 for nb in nbs if k_inf_all[nb] > my_kinf)
                kinf_rank_nb = stronger_count / (len(nbs) + 1e-6)
                max_nb_kinf_val = max((k_inf_all[nb] for nb in nbs), default=0)
                is_peak = 1.0 if my_kinf >= max_nb_kinf_val else 0.0
            else:
                kinf_rank_nb = 0.0
                is_peak = 1.0
            
            # 桥梁得分
            sum_nb_kinf = sum(k_inf_all[nb] for nb in infected_nbs) if infected_nbs else 0
            bridge = (k_inf_all[i] * self.degrees[i]) / (sum_nb_kinf + 1e-6)
            norm_bridge = np.log(bridge + 1) / 10.0
            
            # 中心性
            sg_closeness = closeness.get(i, 0.0) if i in infected_set else 0.0
            raw_ecc = eccentricity.get(i, 0.0) if i in infected_set else 0.0
            sg_eccentricity = 1.0 - (raw_ecc / max_ecc) if max_ecc > 0 else 0.0
            
            # 传播方向
            if len(infected_nbs) > 0 and self.degrees[i] > 0:
                smaller_count = sum(1 for nb in infected_nbs if self.degrees[nb] < self.degrees[i])
                outward_ratio = smaller_count / len(infected_nbs)
            else:
                outward_ratio = 0.0
            
            x_observed[i] = [
                is_infected, norm_deg, norm_inf_count, log_deg,
                norm_i_score, norm_kcore, norm_max_nb_kinf, mismatch,
                kinf_rank_nb, is_peak, norm_bridge,
                sg_closeness, sg_eccentricity, outward_ratio
            ]
        
        # 应用特征消融
        if self.ablation_feature_idx is not None:
            x_observed = self._apply_ablation(x_observed)
        
        return x_observed, torch.FloatTensor(k_inf_all), cc_labels, max_cc_id
    
    def _build_source_features(self, source_indices):
        """
        构建源点态特征 (初始状态)
        
        设计原则:
            - 静态特征 (norm_deg, log_deg): 保持不变
            - 状态标志: 源点=1, 非源点=0
            - 传播相关特征: 设为初始值
        
        Args:
            source_indices: 源点索引
        
        Returns:
            x_source: [N, 14] 源点态特征
        """
        source_set = set(source_indices)
        
        # 源点子图中心性 (初始状态的结构特征)
        if len(source_indices) >= 2:
            closeness, eccentricity, max_ecc = self._compute_subgraph_centralities(source_indices)
            kcore_sub = self._compute_subgraph_kcore(source_indices)
            max_kcore = max(kcore_sub.values(), default=1) or 1
        else:
            closeness = {i: 1.0 for i in source_indices}  # 单点默认中心
            eccentricity = {i: 0.0 for i in source_indices}
            max_ecc = 1.0
            kcore_sub = {i: 0 for i in source_indices}
            max_kcore = 1
        
        x_source = np.zeros((self.num_nodes, 14), dtype=np.float32)
        
        for i in range(self.num_nodes):
            is_source = 1.0 if i in source_set else 0.0
            norm_deg = self.degree_percentile[i]
            log_deg = np.log(self.degrees[i] + 1e-5) / 5.0
            
            # 初始状态: 无传播发生
            norm_inf_count = 0.0  # 无感染邻居
            
            # i_score 仅保留度数部分
            ALPHA = 2.0
            norm_i_score = (ALPHA * np.log(self.degrees[i] + 1e-5)) / 50.0
            
            # 源点子图 k-core
            norm_kcore = (kcore_sub.get(i, 0) / max_kcore) if i in source_set else 0.0
            
            # 无传播, 相关特征为 0
            norm_max_nb_kinf = 0.0
            mismatch = 0.0
            kinf_rank_nb = 0.0
            
            # 源点初始状态是峰值
            is_peak = 1.0 if i in source_set else 0.0
            
            # 全局桥梁得分 (基于拓扑结构)
            norm_bridge = self.global_bridge_scores[i]
            
            # 源点子图中心性
            sg_closeness = closeness.get(i, 0.0) if i in source_set else 0.0
            raw_ecc = eccentricity.get(i, 0.0) if i in source_set else 0.0
            sg_eccentricity = 1.0 - (raw_ecc / max_ecc) if max_ecc > 0 else 0.0
            
            # 初始状态无传播方向
            outward_ratio = 0.0
            
            x_source[i] = [
                is_source, norm_deg, norm_inf_count, log_deg,
                norm_i_score, norm_kcore, norm_max_nb_kinf, mismatch,
                kinf_rank_nb, is_peak, norm_bridge,
                sg_closeness, sg_eccentricity, outward_ratio
            ]
        
        # 应用特征消融
        if self.ablation_feature_idx is not None:
            x_source = self._apply_ablation(x_source)
        
        return x_source
    
    def generate_dataset(self, influ_list, cache_name=None, cache_dir='data', use_gfrr=True):
        """
        生成特征数据集
        
        Args:
            influ_list: 传播影响列表
            cache_name: 缓存文件名
            cache_dir: 缓存目录
            use_gfrr: 是否生成 GFRR 所需的 x_source 特征
        
        Returns:
            dataset: Data 对象列表, 每个包含:
                - x: 观测态特征 [N, 14]
                - x_source: 源点态特征 [N, 14] (仅当 use_gfrr=True)
                - edge_index: 边索引 [2, E]
                - y: 标签 [N]
                - train_mask: 感染节点掩码 [N]
                - k_inf: 感染邻居数 [N]
        """
        # 尝试加载缓存
        if cache_name is not None:
            suffix = '_gfrr_multi' if use_gfrr else '_multi'
            cache_path = os.path.join(cache_dir, f'feature_{cache_name}{suffix}.pt')
            
            if os.path.exists(cache_path):
                print(f"[*] 发现缓存: {cache_path}")
                try:
                    cached_data = torch.load(cache_path)
                    # 多快照模式下 Data 数量≥级联数，检查第一个 Data 是否带有 cascade_id
                    has_multi = len(cached_data) > 0 and hasattr(cached_data[0], 'cascade_id')
                    if has_multi:
                        print(f"[*] 从缓存加载 {len(cached_data)} 个样本（多快照模式）")
                        return cached_data
                    print(f"[!] 缓存格式不匹配（旧版单快照）, 重新计算...")
                except Exception as e:
                    print(f"[!] 缓存加载失败: {e}, 重新计算...")
        
        print(f"[*] 构建 GFRR 特征数据集 (use_gfrr={use_gfrr}, 多快照模式)...")
        dataset = []
        
        # 使用原始邻接矩阵构建边索引（全图）
        adj_coo = sp.coo_matrix(self.adj_matrix)
        edge_index = torch.LongTensor(np.array([adj_coo.row, adj_coo.col]))
        node_degrees = torch.FloatTensor(self.degrees)
        
        for cascade_idx, mat in enumerate(influ_list):
            source_vec = mat[:, 0]
            n_snapshot_cols = mat.shape[1] - 1  # 除列0之外共有 T-1 个快照列

            source_indices = np.where(source_vec == 1)[0]
            if len(source_indices) == 0:
                continue

            # 标签（所有快照共享同一个标签）
            y_np = np.zeros(self.num_nodes, dtype=np.float32)
            y_np[source_indices] = 1.0

            # GFRR 模式：预先计算源点态特征（所有快照共享）
            x_source = None
            if use_gfrr:
                x_source = self._build_source_features(source_indices)

            for t_idx in range(n_snapshot_cols):
                infected_vec = mat[:, t_idx + 1]
                infected_indices = np.where(infected_vec == 1)[0]

                if len(infected_indices) == 0:
                    continue

                # 是否为最终快照（推理时使用）
                is_final = (t_idx == n_snapshot_cols - 1)

                # 构建观测态特征
                x_observed, k_inf_tensor, cc_labels, max_cc_id = self._build_observed_features(infected_indices, source_indices)

                # 感染掩码
                train_mask = torch.zeros(self.num_nodes, dtype=torch.bool)
                train_mask[infected_indices] = True

                data = Data(
                    x=torch.FloatTensor(x_observed),
                    edge_index=edge_index,
                    degrees=node_degrees,
                    y=torch.FloatTensor(y_np),
                    train_mask=train_mask,
                    k_inf=k_inf_tensor,
                    cc_labels=torch.LongTensor(cc_labels),
                    max_cc_id=max_cc_id
                )

                # 标记 cascade_id 和 is_final，训练/推理分流用
                data.cascade_id = cascade_idx
                data.is_final = is_final

                if use_gfrr:
                    data.x_source = torch.FloatTensor(x_source)

                dataset.append(data)
            
            if (cascade_idx + 1) % 100 == 0:
                print(f"    已处理 {cascade_idx + 1}/{len(influ_list)} 个级联...")
        
        print(f"[*] 数据集构建完成, 共 {len(dataset)} 个样本（来自 {len(influ_list)} 个级联）")
        
        # 保存缓存
        if cache_name is not None:
            suffix = '_gfrr_multi' if use_gfrr else '_multi'
            cache_path = os.path.join(cache_dir, f'feature_{cache_name}{suffix}.pt')
            os.makedirs(cache_dir, exist_ok=True)
            try:
                torch.save(dataset, cache_path)
                print(f"[*] 已缓存到: {cache_path}")
            except Exception as e:
                print(f"[!] 缓存保存失败: {e}")
        
        return dataset
