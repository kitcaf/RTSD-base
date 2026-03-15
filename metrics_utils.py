"""
评测指标计算模块
计算源定位任务中的专用指标：MAP、P@K_true、AED
"""

import numpy as np
from scipy.sparse.csgraph import shortest_path


def precompute_shortest_paths(adj_matrix):
    """
    预计算全图最短路径矩阵（无权最短跳数）
    仅需计算一次，后续指标计算直接查表，避免重复耗时。

    参数：
        adj_matrix: numpy 数组或 scipy sparse 矩阵，形状 [N, N]

    返回：
        dist_matrix: numpy 数组，形状 [N, N]，dist[i, j] 为 i 到 j 的最短跳数
    """
    dist = shortest_path(csgraph=adj_matrix, directed=False, unweighted=True, return_predecessors=False)
    return np.asarray(dist)


def calculate_map(y_true_prob, y_true_binary, top_k):
    """
    计算平均精度均值 (Mean Average Precision, MAP)
    map的k的范围是从1 ~ 感染区的节点
    参数：
        y_true_prob: 预测的概率值，numpy数组 (n_nodes,)
        y_true_binary: 真实的二值标签，numpy数组 (n_nodes,)
        top_k: 评估时截断的长度（例如感染节点数）
    
    返回：
        map_score: MAP分数（0-1之间）
    
    说明：
        - 只在前 num_sources 个节点范围内计算
        - 根据概率降序排列，计算精度曲线下的面积
    """
    # 获取预测概率排序的索引
    sorted_indices = np.argsort(-y_true_prob)  # 降序排列
    k = min(int(top_k), len(sorted_indices))
    if k <= 0:
        return 0.0
    
    # 只取前 top_k 个节点进行评估
    top_k_indices = sorted_indices[:k]
    
    # 计算精度-召回曲线
    tp = 0
    ap = 0.0
    
    for idx, node_idx in enumerate(top_k_indices):
        if y_true_binary[node_idx] == 1:
            tp += 1
            # 精度 = 目前为止找到的正样本数 / 当前位置
            precision_at_k = tp / (idx + 1)
            ap += precision_at_k
    
    # 计算平均精度（除以真实源点数）
    num_true_sources = int(np.sum(y_true_binary))
    map_score = ap / num_true_sources if num_true_sources > 0 else 0.0
    
    return float(map_score)


def calculate_precision_at_k(y_true_prob, y_true_binary, k):
    """
    计算 P@K (Precision at K)
    
    参数：
        y_true_prob: 预测的概率值，numpy数组 (n_nodes,)
        y_true_binary: 真实的二值标签，numpy数组 (n_nodes,)
        k: 评估的前K个节点
    
    返回：
        p_at_k: 前K个节点中真实源点的精度
    
    说明：
        - k 等于真实源点的数量
        - P@K = 前K个节点中正样本数 / K
    """
    # 获取预测概率排序的索引
    sorted_indices = np.argsort(-y_true_prob)  # 降序排列
    k = min(int(k), len(sorted_indices))
    if k <= 0:
        return 0.0
    
    # 只取前 k 个节点
    top_k_indices = sorted_indices[:k]
    
    # 计算精度：前k个中有多少个是真实源点
    tp = np.sum(y_true_binary[top_k_indices])
    p_at_k = tp / k if k > 0 else 0.0
    
    return float(p_at_k)


def calculate_aed(y_true_prob, y_true_binary, dist_matrix=None, adj_matrix=None, top_k=None, node_indices=None):
    """
    计算拓扑容错性 - 平均欧氏距离 (Average Euclidean Distance, AED)
    
    参数：
        y_true_prob: 预测的概率值，numpy数组 (n_nodes,)
        y_true_binary: 真实的二值标签，numpy数组 (n_nodes,)
        dist_matrix: 预计算的最短路径矩阵 [N, N]
        adj_matrix: 备用邻接矩阵（若 dist_matrix 未提供时使用，但开销大）
        top_k: 评估截断长度（通常为真实源点数）
        node_indices: 将当前数组位置映射回全图节点ID的索引列表/数组
    
    返回：
        aed_score: 平均欧氏距离（跳数）
    
    说明：
        - 计算预测错误的节点到最近真实源点的最短路径距离
        - 在长度为 num_sources 的区间内进行判断
        - 如果一个预测的节点不是真实源点，则累加其到最近源点的距离
    """
    n = len(y_true_prob)
    k = int(top_k) if top_k is not None else int(np.sum(y_true_binary))
    k = min(k, n)
    if k <= 0:
        return 0.0
    
    # 预测节点（截断至 top_k）
    sorted_indices = np.argsort(-y_true_prob)
    predicted_local = sorted_indices[:k]
    true_local = np.where(y_true_binary == 1)[0]
    
    # 映射到全图节点ID
    if node_indices is not None:
        node_indices = np.asarray(node_indices)
        predicted_nodes = node_indices[predicted_local]
        true_sources = node_indices[true_local]
    else:
        predicted_nodes = predicted_local
        true_sources = true_local
    
    if len(true_sources) == 0:
        return 0.0
    
    # 获取距离矩阵
    if dist_matrix is None:
        if adj_matrix is None:
            return 0.0
        dist_matrix = precompute_shortest_paths(adj_matrix)
    
    total_distance = 0.0
    error_count = 0
    n_nodes = dist_matrix.shape[0]
    true_sources_set = set(int(x) for x in true_sources.tolist())
    
    for pred in predicted_nodes:
        if int(pred) not in true_sources_set:
            error_count += 1
            distances = dist_matrix[int(pred), true_sources]
            finite_dist = distances[np.isfinite(distances)]
            if finite_dist.size == 0:
                min_distance = n_nodes
            else:
                min_distance = float(finite_dist.min())
            total_distance += min_distance
    
    aed_score = total_distance / error_count if error_count > 0 else 0.0
    return float(aed_score)
