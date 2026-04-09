"""
评测指标计算模块
计算源定位任务中的专用指标：MAP、P@K_true、AED
"""

import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import shortest_path


class LazyShortestPathMatrix:
    """
    懒加载最短路矩阵。

    设计动机:
        全图 APSP 在当前数据规模下开销过大，而 AED 只会查询少量行。
        因此改为“按需计算单源最短路 + 行缓存”。
    """

    def __init__(self, adj_matrix):
        self.adj_matrix = adj_matrix.tocsr() if sp.issparse(adj_matrix) else sp.csr_matrix(adj_matrix)
        self.shape = self.adj_matrix.shape
        self._row_cache = {}

    def _get_row(self, row_index: int) -> np.ndarray:
        row_index = int(row_index)
        cached_row = self._row_cache.get(row_index)
        if cached_row is not None:
            return cached_row

        dist_row = shortest_path(
            csgraph=self.adj_matrix,
            directed=False,
            unweighted=True,
            indices=row_index,
            return_predecessors=False,
        )
        cached_row = np.asarray(dist_row, dtype=np.float32)
        self._row_cache[row_index] = cached_row
        return cached_row

    def __getitem__(self, item):
        if not isinstance(item, tuple) or len(item) != 2:
            raise TypeError("LazyShortestPathMatrix only supports dist[row, col] indexing")

        row_selector, col_selector = item
        if np.isscalar(row_selector):
            return self._get_row(int(row_selector))[col_selector]

        row_indices = np.asarray(row_selector, dtype=np.int64).reshape(-1)
        row_values = [self._get_row(int(row_idx))[col_selector] for row_idx in row_indices]
        return np.stack(row_values, axis=0)


def precompute_shortest_paths(adj_matrix):
    """
    构造最短路查询器（无权最短跳数）。

    说明:
        旧实现会在实验启动时一次性计算整张图的 APSP，代价非常高。
        新实现改为懒加载查询器，只在 AED 真正访问某一行距离时才计算。

    参数：
        adj_matrix: numpy 数组或 scipy sparse 矩阵，形状 [N, N]

    返回：
        dist_matrix: 支持 dist[i, j] 访问的懒加载对象
    """
    return LazyShortestPathMatrix(adj_matrix)


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
    sorted_indices = np.argsort(-y_true_prob)
    k = min(int(top_k), len(sorted_indices))
    if k <= 0:
        return 0.0
    
    top_k_indices = sorted_indices[:k]
    
    tp = 0
    ap = 0.0
    
    for idx, node_idx in enumerate(top_k_indices):
        if y_true_binary[node_idx] == 1:
            tp += 1
            precision_at_k = tp / (idx + 1)
            ap += precision_at_k
    
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
    sorted_indices = np.argsort(-y_true_prob)
    k = min(int(k), len(sorted_indices))
    if k <= 0:
        return 0.0
    
    top_k_indices = sorted_indices[:k]
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
        adj_matrix: 备用邻接矩阵（若 dist_matrix 未提供时使用）
        top_k: 评估截断长度（通常为真实源点数）
        node_indices: 将当前数组位置映射回全图节点ID的索引列表/数组
    
    返回：
        aed_score: 平均欧氏距离（跳数）
    """
    n = len(y_true_prob)
    k = int(top_k) if top_k is not None else int(np.sum(y_true_binary))
    k = min(k, n)
    if k <= 0:
        return 0.0
    
    sorted_indices = np.argsort(-y_true_prob)
    predicted_local = sorted_indices[:k]
    true_local = np.where(y_true_binary == 1)[0]
    
    if node_indices is not None:
        node_indices = np.asarray(node_indices)
        predicted_nodes = node_indices[predicted_local]
        true_sources = node_indices[true_local]
    else:
        predicted_nodes = predicted_local
        true_sources = true_local
    
    if len(true_sources) == 0:
        return 0.0
    
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
