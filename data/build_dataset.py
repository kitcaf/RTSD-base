#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
真实数据集预处理脚本（单快照版本）
将真实的传播数据转换为源定位任务所需的.SG格式

修改说明：
- 取消了 MIN_CASCADE_SIZE 的严格限制（仅需大于0）
- 取消了 必须发生扩散（final > seed） 的限制
- 保留了 必须存在至少一个源节点 的限制
- 【新增】Step 2: 构建 adj_sets 用于快速查找
- 【新增】Step 4: 增加 analyze_social_explainability 函数调用进行统计
- 【新增】Step 7: 增加平均源点数、平均级联长度、社交解释力分布的输出
"""

import numpy as np
import pickle
import scipy.sparse as sp
from collections import defaultdict
import os
import sys
import analy_utils

# 添加项目根目录到路径，以便导入SparseGraph类
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from data.sparsegraph import SparseGraph

# ============================================================================
# 配置参数 - 所有可调参数集中在此处
# ============================================================================

# 数据集名称（修改此处来处理不同数据集：android, twitter, christianity, douban等）
DATASET_NAME = 'douban'

# 路径配置
SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))   # 脚本所在目录
DATASET_DIR = os.path.join(SCRIPT_DIR, DATASET_NAME)   # 数据集文件夹
EDGES_FILE = os.path.join(DATASET_DIR, 'edges.txt')   # 边文件
CASCADES_FILE = os.path.join(DATASET_DIR, 'cascades.txt')   # 级联文件
OUTPUT_FILE = os.path.join(SCRIPT_DIR, f'{DATASET_NAME}_25c.SG')   # 输出.SG文件
U2IDX_FILE = os.path.join(DATASET_DIR, 'u2idx.pickle')   # 用户ID映射文件
IDX2U_FILE = os.path.join(DATASET_DIR, 'idx2u.pickle')   # 索引到用户映射文件

# 概率矩阵参数（基于度中心性）
PROB_HIGH_DEGREE = [0.05, 0.05, 0.05, 0.05, 0.05, 0.05]  # 高度节点的概率选项
PROB_LOW_DEGREE = [0.05, 0.05, 0.05, 0.05, 0.05, 0.05]   # 低度节点的概率选项
RANDOM_SEED = 42   # 随机种子，保证可重复性

# 级联处理参数
# 中间快照的时间分位数（不含源点列0）；可自由配置
# 最后一个分位数必须为1.0（100%最终状态），会被作为推理时的观测输入
SNAPSHOT_QUANTILES = [1.0]   # 可配置：例如改为 [0.5, 1.0] 只取两个快照
NUM_TIMESTEPS = 1 + len(SNAPSHOT_QUANTILES)     # 自动计算：列0=源点，列1..T-1=各时间快照
SOURCE_RATIO = 0.05  # 源节点时间比例（前5%时间内出现的节点作为源节点）

# 【修改点1】将最小级联大小改为1，仅为了防止空数据报错，实际上取消了长度筛选
MIN_CASCADE_SIZE = 1  
MIN_SOURCE_NODES = 1  # 最小源节点数
MAX_SAMPLES = None  # 最大样本数（None表示不限制）

# 图类型参数
DIRECTED_GRAPH = False  # 是否为有向图（False表示无向图，与karate保持一致）

# 【新增】源点筛选参数
REMOVE_ZERO_KINF_SOURCES = True  # 是否移除局部感染度(k_inf)为0的源点
                                   # True: 移除那些没有感染邻居的源点（孤立源点）
                                   # False: 保留所有源点（默认行为）

# ============================================================================
# 脚本开始
# ============================================================================

print("=" * 70)
print(f"{DATASET_NAME.upper()} 数据集预处理脚本")
print("=" * 70)
print(f"\n配置参数:")
print(f"  - 数据集名称: {DATASET_NAME}")
print(f"  - 数据集目录: {DATASET_DIR}")
print(f"  - 输出文件: {OUTPUT_FILE}")
print(f"  - 时间步数: {NUM_TIMESTEPS} (列0=源点 + {len(SNAPSHOT_QUANTILES)}个快照)")
print(f"  - 快照分位数: {SNAPSHOT_QUANTILES}")
print(f"  - 源节点时间比例: 前5%")
print(f"  - 最小源节点数: {MIN_SOURCE_NODES}")
print(f"  - 移除k_inf=0的源点: {REMOVE_ZERO_KINF_SOURCES}")
print(f"  - 筛选策略: 已取消级联长度限制，已取消扩散行为限制")

# ============================================================================
# 步骤1: 读取边文件，构建用户ID映射
# ============================================================================
print("\n[步骤1] 读取网络拓扑结构...")

unique_users = set()

# 读取所有边，收集所有唯一用户
edges_list = []
with open(EDGES_FILE, 'r') as f:
    for line in f:
        line = line.strip()
        if line:
            parts = line.split(',')
            if len(parts) == 2:
                user1, user2 = parts[0].strip(), parts[1].strip()
                unique_users.add(user1)
                unique_users.add(user2)
                edges_list.append((user1, user2))

print(f"  - 读取社交边数: {len(edges_list)}")
print(f"  - 唯一用户数: {len(unique_users)}")

# 构建用户ID到索引的映射 (从0开始)
sorted_users = sorted(list(unique_users), key=lambda x: int(x))
u2idx = {user: idx for idx, user in enumerate(sorted_users)}
idx2u = {idx: user for user, idx in u2idx.items()}

print(f"  - 用户ID映射完成: {len(u2idx)} 个用户")

# 保存映射关系
with open(U2IDX_FILE, 'wb') as f:
    pickle.dump(u2idx, f)
with open(IDX2U_FILE, 'wb') as f:
    pickle.dump(idx2u, f)

# ============================================================================
# 步骤2: 构建邻接矩阵
# ============================================================================
print("\n[步骤2] 构建邻接矩阵...")

n_nodes = len(u2idx)
adj_matrix = np.zeros((n_nodes, n_nodes), dtype=np.float32)

# 【新增】构建邻接表 (Set格式) 用于 Step 4 的快速统计
adj_sets = defaultdict(set)

# 构建邻接矩阵 (有向图或无向图)
for user1, user2 in edges_list:
    idx1, idx2 = u2idx[user1], u2idx[user2]
    adj_matrix[idx1, idx2] = 1
    
    # 【新增】同时构建邻接表
    adj_sets[idx1].add(idx2)
    
    if not DIRECTED_GRAPH:
        adj_matrix[idx2, idx1] = 1  # 无向图需要对称
        adj_sets[idx2].add(idx1)    # 【新增】

# 转换为稀疏矩阵
adj_sparse = sp.csr_matrix(adj_matrix)
print(f"  - 邻接矩阵形状: {adj_sparse.shape}")
print(f"  - 非零边数: {adj_sparse.nnz}")

# ============================================================================
# 步骤4: 处理级联数据，构建训练样本
# ============================================================================
print("\n[步骤4] 处理传播级联数据...")

all_samples = []
valid_cascades = 0
skipped_cascades = 0

# 【新增】全局统计变量
stats_global_targets = 0
stats_global_hop1 = 0
stats_global_hop2 = 0
stats_global_hop3plus = 0
stats_global_hop1_from_source = 0
stats_global_hop2_depend_source = 0
stats_global_zero_infect = 0

with open(CASCADES_FILE, 'r') as f:
    errorCasLine = 0
    errorNode = 0
    for line_num, line in enumerate(f, 1):
        line = line.strip()
        if not line:
            continue
        
        # 解析级联数据
        chunks = line.split(',')
        userlist = []
        timestamplist = []
       
        for chunk in chunks:
            chunk = chunk.strip()
            flag = False
            if not chunk:
                continue
            
            try:
                parts = chunk.split()
                if len(parts) == 2:
                    user, timestamp = parts
                elif len(parts) == 3:
                    root, user, timestamp = parts
                    # 添加root节点
                    if root in u2idx:
                        userlist.append(u2idx[root])
                        timestamplist.append(float(timestamp))
                else:
                    continue
                
                # 添加当前用户节点
                if user in u2idx:
                    userlist.append(u2idx[user])
                    timestamplist.append(float(timestamp))
                else: 
                    flag = True
                    errorNode += 1
            except Exception as e:
                continue
            if flag:
                errorCasLine += 1
        
        # 【修改点2】移除原来的 MIN_CASCADE_SIZE 检查
        # 只要列表不为空即可继续，防止min()函数报错
        if not userlist:
            skipped_cascades += 1
            continue
        
        # 去重，保留第一次出现的节点
        # 【注意】这里必须保持时间顺序，才能正确统计 Hop 分布
        # 原始代码是: zip -> seen -> append, 这依赖于输入本来就是大致有序的
        # 为了更严谨，我们先按时间排序一下，再进行去重逻辑
        
        # 将 userlist 和 timestamplist 组合并按时间排序
        combined = sorted(zip(userlist, timestamplist), key=lambda x: x[1])
        
        seen_users = set()
        unique_userlist = []
        unique_timestamplist = []
        
        for user, timestamp in combined:
            if user not in seen_users:
                seen_users.add(user)
                unique_userlist.append(user)
                unique_timestamplist.append(timestamp)
        
        userlist = unique_userlist
        timestamplist = unique_timestamplist
        
        # 【修改点3】移除去重后的 MIN_CASCADE_SIZE 检查
        if not userlist:
            skipped_cascades += 1
            continue
        
        # 归一化时间戳到[0, 1]范围
        min_timestamp = min(timestamplist)
        max_timestamp = max(timestamplist)
        time_range = max_timestamp - min_timestamp
        
        if time_range > 0:
            # 归一化时间戳
            normalized_timestamps = [(ts - min_timestamp) / time_range for ts in timestamplist]
        else:
            # 如果所有节点时间戳相同，均匀分配
            normalized_timestamps = [i / len(timestamplist) for i in range(len(timestamplist))]
        
        # 构建时间序列影响力矩阵 (N, T)
        # 列0:   源节点快照（标签目标）
        # 列1~T-2: 中间时刻快照（训练时的额外观测）
        # 列T-1:  最终扩散快照（推理时的唯一观测输入）
        influ_mat = np.zeros((n_nodes, NUM_TIMESTEPS), dtype=np.float32)
        
        # 确定源节点：前5%时间内出现的节点作为t=0的种子节点
        source_time_threshold = SOURCE_RATIO  # 5%的时间
        source_nodes = [user for user, norm_ts in zip(userlist, normalized_timestamps) 
                       if norm_ts <= source_time_threshold]
        
        # 确保至少有MIN_SOURCE_NODES个源节点
        if len(source_nodes) < MIN_SOURCE_NODES:
            source_nodes = userlist[:MIN_SOURCE_NODES]
        
        # 【新增】如果开启了REMOVE_ZERO_KINF_SOURCES，移除局部感染度为0的源点
        if REMOVE_ZERO_KINF_SOURCES:
            # 计算每个源点的局部感染度 k_inf（感染邻居数）
            infected_set = set(userlist)  # 所有感染节点
            filtered_sources = []
            for src in source_nodes:
                # 计算该源点的感染邻居数
                k_inf = sum(1 for nb in adj_sets.get(src, set()) if nb in infected_set)
                if k_inf > 0:
                    filtered_sources.append(src)
                else:
                    stats_global_zero_infect += 1
            
            # 如果过滤后源点数不足MIN_SOURCE_NODES，跳过该级联
            # 因为k_inf=0的源点对训练没有价值
            if len(filtered_sources) < MIN_SOURCE_NODES:
                skipped_cascades += 1
                continue
            
            source_nodes = filtered_sources
        
        # 列0: 设置源节点（目标快照）
        for node in source_nodes:
            influ_mat[node, 0] = 1.0
        
        # 列1~T-1: 按 SNAPSHOT_QUANTILES 切出各时刻快照
        # 已按时间排序的 userlist / normalized_timestamps 直接用于筛选
        for t_idx, q in enumerate(SNAPSHOT_QUANTILES):
            col = t_idx + 1  # 对应 influ_mat 的第 col 列
            for user, norm_ts in zip(userlist, normalized_timestamps):
                if norm_ts <= q:
                    influ_mat[user, col] = 1.0
        
        # 检查是否有效（以最终列 T-1 为准）
        seed_count = influ_mat[:, 0].sum()
        final_count = influ_mat[:, -1].sum()
        
        # 【修改点4】修改判断条件
        if seed_count > 0:
            # 【新增】在此处调用函数进行统计分析
            # 需要传入：邻接表，排序好的用户列表，源节点ID集合
            source_set = set(source_nodes)
            (t_targets, t_h1, t_h2, t_h3, 
             t_h1_src, t_h2_dep) = analy_utils.analyze_social_explainability(adj_sets, userlist, source_set)
            
            # 累加到全局
            stats_global_targets += t_targets
            stats_global_hop1 += t_h1
            stats_global_hop2 += t_h2
            stats_global_hop3plus += t_h3
            stats_global_hop1_from_source += t_h1_src
            stats_global_hop2_depend_source += t_h2_dep
            
            # 加入样本
            all_samples.append(influ_mat)
            valid_cascades += 1
            
            # 限制最大样本数
            if MAX_SAMPLES is not None and valid_cascades >= MAX_SAMPLES:
                break
            
    print(f"该数据集出现{errorCasLine}个级联里面出现节点未知")
    print(f"该数据集出现{errorNode}个未知节点")

print(f"  - 总级联数: {line_num}")
print(f"  - 有效级联数: {valid_cascades}")
print(f"  - 跳过级联数: {skipped_cascades} (仅跳过空级联)")

if valid_cascades == 0:
    print("\n错误: 没有有效的级联数据!")
    exit(1)

# ============================================================================
# 步骤5: 构建influ_mat_list张量
# ============================================================================
print("\n[步骤5] 构建训练样本张量...")

# 将所有样本堆叠成三维数组: [M, N, T]
influ_mat_list = np.array(all_samples, dtype=np.float32)  # [M, N, T]

print(f"  - influ_mat_list形状: {influ_mat_list.shape}")
print(f"  - 样本数 (M): {influ_mat_list.shape[0]}")
print(f"  - 节点数 (N): {influ_mat_list.shape[1]}")
print(f"  - 时间步数 (T): {influ_mat_list.shape[2]}")

# 验证数据的有效性
print("\n  样本统计:")
for i in range(min(3, len(all_samples))):
    seed_count = int(influ_mat_list[i, :, 0].sum())
    final_count = int(influ_mat_list[i, :, -1].sum())
    print(f"    样本 #{i}: 源节点数={seed_count}, 最终影响节点数={final_count}")


# ============================================================================
# 步骤6: 保存为.SG文件
# ============================================================================
print("\n[步骤6] 保存为.SG文件...")

# 构建SparseGraph对象（只传入adj_matrix）
graph = SparseGraph(adj_sparse)

# 手动设置influ_mat_list属性
graph.influ_mat_list = influ_mat_list

with open(OUTPUT_FILE, 'wb') as f:
    pickle.dump(graph, f)

print(f"  - 数据已保存到: {OUTPUT_FILE}")


# ============================================================================
# 验证数据
# ============================================================================
print("\n[步骤7] 验证生成的数据...")

with open(OUTPUT_FILE, 'rb') as f:
    loaded_graph = pickle.load(f)

# 从加载的数据中获取变量
n_nodes = loaded_graph.adj_matrix.shape[0]
influ_mat_list = loaded_graph.influ_mat_list
n_samples = influ_mat_list.shape[0]

# 获取最后一个时间步的感染状态 (M, N)
final_state_matrix = influ_mat_list[:, :, -1]

# 计算每个样本的最终感染节点数
infected_counts_per_sample = np.sum(final_state_matrix, axis=1)

# 计算平均值
avg_infected_nodes = np.mean(infected_counts_per_sample)
avg_infected_proportion = avg_infected_nodes / n_nodes

avg_uninfected_nodes = n_nodes - avg_infected_nodes
avg_uninfected_proportion = 1.0 - avg_infected_proportion

# 【新增】计算平均源点数
source_counts_per_sample = np.sum(influ_mat_list[:, :, 0], axis=1)
avg_source_count = np.mean(source_counts_per_sample)

print(f"  - 节点总数 (N): {n_nodes}")
print(f"  - 样本总数 (M): {n_samples}")
print(f"  - [指标] 平均每个样本的感染节点数 (Avg Cascade Length): {avg_infected_nodes:.4f}")
print(f"  - [指标] 平均每个样本的源点数 (Avg Source Count): {avg_source_count:.4f}")
print(f"  - 平均每个样本的感染节点比例: {avg_infected_proportion * 100:.2f}%")
print(f"  - 平均每个样本的未感染节点数: {avg_uninfected_nodes:.2f}")
print(f"  - 平均每个样本的未感染节点比例: {avg_uninfected_proportion * 100:.2f}%")
print(f"  - 累计删除的源点数 (k_inf=0，无感染邻居的孤立源点): {stats_global_zero_infect}")

# 【新增】输出社交解释力分布结果
print("\n" + "-" * 50)
print("【社交网络解释力分析 (Social Explanatory Power)】")
if stats_global_targets > 0:
    r1 = stats_global_hop1 / stats_global_targets
    r2 = stats_global_hop2 / stats_global_targets
    r3 = stats_global_hop3plus / stats_global_targets
    
    # 源点依赖覆盖率 (分母是各自Hop类别的节点数)
    r_h1_src = stats_global_hop1_from_source / stats_global_hop1 if stats_global_hop1 > 0 else 0
    r_h2_dep = stats_global_hop2_depend_source / stats_global_hop2 if stats_global_hop2 > 0 else 0
    
    print(f"  感染非源点总数 (Total Targets): {stats_global_targets}")
    print(f"  [Hop-1] 显式覆盖率 (Direct Friends):     {stats_global_hop1:6d} ({r1:.2%})")
    print(f"          其中前驱为源点数:                {stats_global_hop1_from_source:6d} ({r_h1_src:.2%})")
    print(f"  [Hop-2] 隐式覆盖率 (Friends of Friends): {stats_global_hop2:6d} ({r2:.2%})")
    print(f"          其中依赖源点数:                  {stats_global_hop2_depend_source:6d} ({r_h2_dep:.2%})")
    print(f"  [Hop-3+] 断层/长尾 (Non-Social/Gap):     {stats_global_hop3plus:6d} ({r3:.2%})")
else:
    print("  无有效的传播目标事件，无法计算分布。")

print("\n" + "=" * 70)
print("数据集构建完成!")
print("=" * 70)