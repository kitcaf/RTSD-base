#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PPR Baseline Source Localization Script (Corrected Version)
基于 Personalized PageRank 的源点定位基准测试

修正点：
1. 严格对齐预处理脚本生成的 .SG 文件结构 (SparseGraph 对象)
2. 正确读取 influ_mat_list 的维度: [:, :, 0]为源点, [:, :, 1]为感染快照
3. 实现了基于 PPR 的反向定位逻辑

传统的中心性算法（PPR）在 Top-1 上完全失效（2.47%），
因为它无法区分 Hub 和 Source。因此，我们需要引入深度学
习模型来学习更复杂的判别模式。
"""

import numpy as np
import pickle
import scipy.sparse as sp
import os
import sys
import time

# ============================================================================
# 路径与环境设置 (与您的预处理脚本保持一致)
# ============================================================================
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

# 尝试导入 SparseGraph，如果 pickle 加载时找不到类定义会报错
try:
    from data.sparsegraph import SparseGraph
except ImportError:
    print("错误: 无法导入 data.sparsegraph.SparseGraph。请确保脚本位置正确。")
    # 为了防止 IDE 报错，定义一个伪类 (实际运行时必须加载真实的类)
    class SparseGraph:
        def __init__(self, adj_matrix):
            self.adj_matrix = adj_matrix
            self.influ_mat_list = None

# ============================================================================
# 配置参数
# ============================================================================

DATASET_NAME = 'twitter'  # 修改这里切换数据集: android, douban, twitter, christianity
ALPHA = 0.85             # 阻尼系数 (1-Alpha 为重启概率)
MAX_ITER = 100           # PPR 迭代次数
TOP_K_METRICS = [1, 5, 10, 20, 50, 100] # 评估指标 K 值

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
# 对应预处理脚本输出的文件名
INPUT_FILE = os.path.join(SCRIPT_DIR, f'{DATASET_NAME}_25c.SG')

# ============================================================================
# 核心函数: Sparse Personalized PageRank
# ============================================================================

def calc_ppr_scipy(adj_norm, seeds, alpha=0.85, max_iter=50, tol=1e-6):
    """
    使用 Scipy 稀疏矩阵进行 PPR 计算
    逻辑：PPR向量 r = alpha * A.T * r + (1-alpha) * p
    其中 p 是个性化向量 (在种子节点处有值)
    """
    n_nodes = adj_norm.shape[0]
    
    # 1. 构建个性化向量 p (Personalization Vector)
    # 在所有感染节点(seeds)处均匀分布概率
    p = np.zeros(n_nodes)
    if len(seeds) > 0:
        p[seeds] = 1.0 / len(seeds)
    else:
        # 兜底：如果没有种子，均匀分布全图
        p[:] = 1.0 / n_nodes
        
    # 2. 初始化 r
    r = p.copy()
    
    # 3. 幂迭代 (Power Iteration)
    # 注意：如果 adj_norm 是行归一化的 (D^-1 A)，传播时应用 A^T
    adj_t = adj_norm.transpose()
    
    for i in range(max_iter):
        r_old = r
        # 迭代公式: r(t+1) = alpha * M^T * r(t) + (1-alpha) * p
        # 这里 alpha=0.85 表示保留概率，(1-alpha) 表示跳回概率
        r = alpha * (adj_t.dot(r_old)) + (1 - alpha) * p
        
        # 检查收敛
        err = np.sum(np.abs(r - r_old))
        if err < tol:
            break
            
    return r

# ============================================================================
# 主流程
# ============================================================================

def main():
    print("=" * 70)
    print(f"PPR Baseline Evaluation: {DATASET_NAME}")
    print("=" * 70)

    # 1. 加载数据
    print(f"[1] 读取 .SG 文件: {INPUT_FILE}")
    if not os.path.exists(INPUT_FILE):
        print(f"错误: 文件 {INPUT_FILE} 不存在！请先运行预处理脚本。")
        return

    with open(INPUT_FILE, 'rb') as f:
        graph = pickle.load(f)
    
    # 从 SparseGraph 对象中提取数据
    adj_matrix = graph.adj_matrix
    # influ_mat_list 形状: [Samples, Nodes, Timesteps=2]
    influ_mat_list = graph.influ_mat_list 
    
    n_nodes = adj_matrix.shape[0]
    n_samples = influ_mat_list.shape[0]
    
    print(f"  - 节点总数: {n_nodes}")
    print(f"  - 样本总数: {n_samples}")
    print(f"  - 数据形状: {influ_mat_list.shape}")

    # 2. 预处理图结构 (行归一化)
    print("\n[2] 构建归一化邻接矩阵...")
    # 计算出度 (Row Sum)
    degree = np.array(adj_matrix.sum(axis=1)).flatten()
    # 防止除以0，孤立点度数设为1 (它们不会传播，所以不影响)
    degree[degree == 0] = 1.0 
    
    # 构建 D^-1
    d_inv = sp.diags(1.0 / degree)
    # P = D^-1 * A (行和为1的转移矩阵)
    adj_norm = d_inv.dot(adj_matrix)
    
    print(f"  - 归一化完成，矩阵类型: {type(adj_norm)}")

    # 3. 运行评估
    print(f"\n[3] 开始 PPR 预测 (Alpha={ALPHA})...")
    
    # 初始化指标统计
    metrics = {k: 0 for k in TOP_K_METRICS}
    metrics['mrr'] = 0.0
    metrics['rank_sum'] = 0
    
    start_time = time.time()
    valid_eval_samples = 0
    
    for i in range(n_samples):
        # --- 提取当前样本数据 ---
        # T=0 是 Label (Source), T=1 是 Input (Snapshot)
        label_vec = influ_mat_list[i, :, 0]
        input_vec = influ_mat_list[i, :, 1]
        
        # 获取索引
        true_source_indices = np.where(label_vec > 0)[0] # 真实源点
        infected_indices = np.where(input_vec > 0)[0]    # 感染快照 (作为PPR种子)
        
        # 异常检查
        if len(true_source_indices) == 0 or len(infected_indices) == 0:
            continue
            
        # --- 核心算法: 运行 PPR ---
        # 用所有感染节点作为"重启集"，寻找这些节点的"结构中心"
        ppr_scores = calc_ppr_scipy(adj_norm, infected_indices, alpha=ALPHA, max_iter=MAX_ITER)
        
        # --- 预测排序 ---
        # argsort是从小到大，取负号或[::-1]变为从大到小
        # 我们需要分数最高的节点排在前面
        ranked_nodes = np.argsort(ppr_scores)[::-1]
        
        # --- 计算指标 ---
        # 目标：找到任意一个真实源点在预测列表中的排名 (Optimistic Rank)
        # 只要预测出的 Top-K 里包含至少一个真实源点，就算命中
        
        # 为了加速，我们把真实源点转为 Set
        true_source_set = set(true_source_indices)
        
        # 找到第一个命中的排名
        first_hit_rank = -1
        for rank, node_idx in enumerate(ranked_nodes):
            if node_idx in true_source_set:
                first_hit_rank = rank + 1 # 排名从1开始
                break
        
        if first_hit_rank == -1:
            # 理论上不会发生，除非源点不在图中
            first_hit_rank = n_nodes
            
        # 累加指标
        for k in TOP_K_METRICS:
            if first_hit_rank <= k:
                metrics[k] += 1
                
        metrics['rank_sum'] += first_hit_rank
        metrics['mrr'] += 1.0 / first_hit_rank
        
        valid_eval_samples += 1
        
        # 打印进度
        if (i + 1) % 100 == 0:
            elapsed = time.time() - start_time
            acc_10 = metrics[10] / valid_eval_samples
            print(f"  - 进度: {i+1}/{n_samples} | 耗时: {elapsed:.1f}s | 当前 Acc@10: {acc_10:.2%}")

    # 4. 输出最终报告
    print("\n" + "=" * 70)
    print(f"PPR Baseline 最终结果 (样本数: {valid_eval_samples})")
    print("=" * 70)
    
    print(f"{'Metric':<15} | {'Value':<10}")
    print("-" * 30)
    
    for k in TOP_K_METRICS:
        acc = metrics[k] / valid_eval_samples
        print(f"Accuracy @ {k:<4} | {acc:.2%}")
        
    print("-" * 30)
    print(f"Mean Rank      | {metrics['rank_sum'] / valid_eval_samples:.2f}")
    print(f"MRR            | {metrics['mrr'] / valid_eval_samples:.4f}")
    print("=" * 70)

if __name__ == "__main__":
    main()