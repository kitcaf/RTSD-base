"""
社交图上的感染子图连通性诊断脚本

对每个级联，在社交图上提取感染节点的诱导子图，分析：
1. 感染子图有多少个连通分量
2. 每个连通分量中是否包含源点
3. 无源点的连通分量（"孤岛"）的规模与占比
4. 源点分布在几个不同的连通分量中
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import networkx as nx
from collections import Counter, defaultdict
from data_loader import load_raw_data


def _fmt_table_value(value, is_percent=False):
    if value is None:
        return "N/A"
    return f"{value:.1f}%" if is_percent else f"{value:.2f}"


def analyze_infected_subgraph(dataset_name='douban_25c.SG', source_ratio=0.05, min_sources=2, max_cascades=None):
    adj, influ = load_raw_data(dataset_name)
    N = adj.shape[0]
    M = len(influ)  # 级联数

    # 构建社交图
    G_social = nx.from_numpy_array(adj)
    print(f"社交图: {N} 节点, {G_social.number_of_edges()} 边")
    print(f"级联数: {M}")
    print()

    # ===== 统计容器 =====
    stats_num_components = []        # 每个级联的连通分量数
    stats_orphan_components = []     # 无源点连通分量数
    stats_orphan_nodes = []          # 无源点连通分量的总节点数
    stats_source_spread = []         # 源点分布在几个连通分量中
    stats_infected_size = []
    stats_source_count = []
    stats_isolated_sources = []      # 孤立源点数 (没有感染邻居的源点)
    stats_max_component_ratio = []   # 最大连通分量占比

    # 连通分量大小分布
    component_sizes_all = []
    orphan_cc_sizes_all = []         # 每个孤岛CC的大小（单独记录）
    source_cc_unified_sizes = []     # 源点100%聚集在同一CC时，该CC的大小
    unified_cascade_all_cc_sizes = [] # 源点100%聚集在同一CC的那些级联中，所有CC的大小
    unified_source_cc_is_largest = [] # 源点100%聚集时，该源点CC是否为本级联最大CC

    cascades_to_process = min(M, max_cascades) if max_cascades else M


    for i in range(cascades_to_process):
        cascade = influ[i]  # [N, T]: T=0 源点, T=1~3 中间快照, T=-1 最终状态

        source_nodes = set(np.where(cascade[:, 0] == 1)[0].tolist())
        infected_nodes = set(np.where(cascade[:, -1] == 1)[0].tolist())  # 取最后一列=100%最终观测

        # 如果源点不在感染节点中,加入（源点本身也是感染的）
        all_infected = infected_nodes | source_nodes

        if len(all_infected) < 2:
            continue

        stats_infected_size.append(len(all_infected))
        stats_source_count.append(len(source_nodes))

        # 构建感染诱导子图（只在感染节点之间保留社交边）
        G_sub = G_social.subgraph(all_infected).copy()

        # 连通分量分析
        components = list(nx.connected_components(G_sub))
        num_cc = len(components)
        stats_num_components.append(num_cc)

        # 按大小排序（降序）
        components.sort(key=len, reverse=True)
        stats_max_component_ratio.append(len(components[0]) / len(all_infected))

        # 分析每个连通分量
        orphan_cc_count = 0
        orphan_node_count = 0
        source_cc_count = 0
        isolated_src = 0

        source_cc_sizes_local = []
        cascade_cc_sizes = []  # 本级联所有CC大小（临时）
        for cc in components:
            component_sizes_all.append(len(cc))
            cascade_cc_sizes.append(len(cc))
            sources_in_cc = cc & source_nodes
            if len(sources_in_cc) > 0:
                source_cc_count += 1
                source_cc_sizes_local.append(len(cc))
            else:
                orphan_cc_count += 1
                orphan_node_count += len(cc)
                orphan_cc_sizes_all.append(len(cc))

        if source_cc_count == 1:
            unified_size = source_cc_sizes_local[0]
            source_cc_unified_sizes.append(unified_size)
            unified_cascade_all_cc_sizes.extend(cascade_cc_sizes)
            max_cc_size = len(components[0])
            unified_source_cc_is_largest.append(unified_size == max_cc_size)

        # 孤立源点: 源点在感染子图中度为0
        for s in source_nodes:
            if G_sub.degree(s) == 0:
                isolated_src += 1

        stats_orphan_components.append(orphan_cc_count)
        stats_orphan_nodes.append(orphan_node_count)
        stats_source_spread.append(source_cc_count)
        stats_isolated_sources.append(isolated_src)

    # ===== 输出报告 =====
    print("=" * 70)
    print(f"  感染子图连通性分析: {dataset_name}")
    print("=" * 70)
    print()

    total = len(stats_num_components)
    print(f"有效级联数: {total}")
    print(f"平均感染节点数: {np.mean(stats_infected_size):.2f}")
    print(f"平均源点数: {np.mean(stats_source_count):.2f}")
    print()

    # 1. 连通分量数分布
    cc_counts = Counter(stats_num_components)
    print("--- 1. 感染子图连通分量数分布 ---")
    print(f"  平均连通分量数: {np.mean(stats_num_components):.2f} ± {np.std(stats_num_components):.2f}")
    print(f"  中位数: {np.median(stats_num_components):.0f}")
    print(f"  最大值: {max(stats_num_components)}")
    for k in sorted(cc_counts.keys()):
        pct = cc_counts[k] / total * 100
        bar = '#' * int(pct / 2)
        print(f"  CC={k:>3d}: {cc_counts[k]:>5d} ({pct:5.1f}%) {bar}")
        if k > 15:
            remaining = sum(v for kk, v in cc_counts.items() if kk > k)
            print(f"  CC>{k:>3d}: {remaining:>5d} ({remaining/total*100:5.1f}%)")
            break
    print()

    # 2. 无源点连通分量（孤岛）
    has_orphan = sum(1 for x in stats_orphan_components if x > 0)
    print("--- 2. 无源点连通分量（'孤岛'） ---")
    print(f"  存在孤岛的级联比例: {has_orphan}/{total} = {has_orphan/total*100:.1f}%")
    print(f"  平均孤岛连通分量数: {np.mean(stats_orphan_components):.2f}")
    if has_orphan > 0:
        orphan_only = [x for x in stats_orphan_components if x > 0]
        print(f"  (仅有孤岛的级联) 平均孤岛数: {np.mean(orphan_only):.2f}")
    orphan_node_ratios = [
        stats_orphan_nodes[i] / stats_infected_size[i]
        for i in range(total) if stats_orphan_components[i] > 0
    ]
    if orphan_node_ratios:
        print(f"  孤岛节点占感染节点的平均比例: {np.mean(orphan_node_ratios)*100:.2f}%")
    if orphan_cc_sizes_all:
        arr = np.array(orphan_cc_sizes_all)
        print(f"  孤岛CC平均大小: {np.mean(arr):.2f} ± {np.std(arr):.2f}")
        print(f"  孤岛CC大小中位数: {np.median(arr):.0f}")
    print()

    # 3. 源点分散度
    print("--- 3. 源点在连通分量中的分散度 ---")
    print(f"  源点分布的连通分量数: {np.mean(stats_source_spread):.2f} ± {np.std(stats_source_spread):.2f}")
    spread_counts = Counter(stats_source_spread)
    for k in sorted(spread_counts.keys()):
        pct = spread_counts[k] / total * 100
        print(f"  源点在 {k} 个CC中: {spread_counts[k]:>5d} ({pct:5.1f}%)")
        if k > 10:
            break
    if source_cc_unified_sizes:
        arr = np.array(source_cc_unified_sizes)
        unified_pct = len(source_cc_unified_sizes) / total * 100
        print(f"  源点100%聚集在同一CC的级联数: {len(source_cc_unified_sizes)}/{total} ({unified_pct:.1f}%)")
        print(f"  该含源点CC平均大小: {np.mean(arr):.2f} ± {np.std(arr):.2f}")
        unified_infected = [stats_infected_size[i] for i in range(total) if stats_source_spread[i] == 1]
        if unified_infected:
            ratios = arr / np.array(unified_infected)
            print(f"  该含源点CC占感染节点平均比例: {np.mean(ratios)*100:.2f}%")
    if unified_cascade_all_cc_sizes:
        arr2 = np.array(unified_cascade_all_cc_sizes)
        print(f"  (这些级联)所有CC平均大小: {np.mean(arr2):.2f} ± {np.std(arr2):.2f}")
        print(f"  (这些级联)所有CC大小中位数: {np.median(arr2):.0f}")
    print()

    # 4. 孤立源点 (无感染邻居直连)
    has_iso = sum(1 for x in stats_isolated_sources if x > 0)
    print("--- 4. 孤立源点 (在感染子图中度=0) ---")
    print(f"  存在孤立源点的级联比例: {has_iso}/{total} = {has_iso/total*100:.1f}%")
    if has_iso > 0:
        iso_only = [x for x in stats_isolated_sources if x > 0]
        print(f"  (有孤立源点的级联) 平均孤立源点数: {np.mean(iso_only):.2f}")
        iso_ratios = [
            stats_isolated_sources[i] / stats_source_count[i]
            for i in range(total) if stats_isolated_sources[i] > 0
        ]
        print(f"  孤立源点占总源点的平均比例: {np.mean(iso_ratios)*100:.2f}%")
    print()

    # 5. 最大连通分量占比
    print("--- 5. 最大连通分量占比 ---")
    print(f"  平均最大CC占比: {np.mean(stats_max_component_ratio)*100:.2f}%")
    print(f"  中位数: {np.median(stats_max_component_ratio)*100:.2f}%")
    bins = [0, 0.5, 0.7, 0.9, 1.0, 1.01]
    labels = ['<50%', '50-70%', '70-90%', '90-100%', '100%']
    for lo, hi, lab in zip(bins[:-1], bins[1:], labels):
        cnt = sum(1 for x in stats_max_component_ratio if lo <= x < hi)
        # 100% 需要特殊处理
        if lab == '100%':
            cnt = sum(1 for x in stats_max_component_ratio if x >= 1.0 - 1e-9)
        print(f"  {lab:>8s}: {cnt:>5d} ({cnt/total*100:5.1f}%)")
    print()

    # 6. 连通分量大小分布
    print("--- 6. 连通分量大小分布 (所有级联汇总) ---")
    size_arr = np.array(component_sizes_all)
    print(f"  总连通分量数: {len(size_arr)}")
    print(f"  大小 [Min, 25%, 50%, 75%, Max]: [{np.min(size_arr)}, {np.percentile(size_arr,25):.0f}, "
          f"{np.median(size_arr):.0f}, {np.percentile(size_arr,75):.0f}, {np.max(size_arr)}]")
    size_bins = [(1, 1), (2, 5), (6, 10), (11, 20), (21, 50), (51, None)]
    for lo, hi in size_bins:
        if hi is None:
            cnt = np.sum(size_arr >= lo)
            label = f">={lo}"
        else:
            cnt = np.sum((size_arr >= lo) & (size_arr <= hi))
            label = f"{lo}-{hi}" if lo != hi else str(lo)
        print(f"  大小 {label:>6s}: {cnt:>6d} ({cnt/len(size_arr)*100:5.1f}%)")
    print()

    # 7. 对比: 连通 vs 不连通级联的源点特征
    print("--- 7. 连通 vs 非连通级联对比 ---")
    connected_idx = [i for i in range(total) if stats_num_components[i] == 1]
    disconnected_idx = [i for i in range(total) if stats_num_components[i] > 1]
    print(f"  完全连通的级联: {len(connected_idx)} ({len(connected_idx)/total*100:.1f}%)")
    print(f"  非连通的级联: {len(disconnected_idx)} ({len(disconnected_idx)/total*100:.1f}%)")
    if connected_idx:
        print(f"  连通级联平均感染数: {np.mean([stats_infected_size[i] for i in connected_idx]):.2f}")
        print(f"  连通级联平均源点数: {np.mean([stats_source_count[i] for i in connected_idx]):.2f}")
    if disconnected_idx:
        print(f"  非连通级联平均感染数: {np.mean([stats_infected_size[i] for i in disconnected_idx]):.2f}")
        print(f"  非连通级联平均源点数: {np.mean([stats_source_count[i] for i in disconnected_idx]):.2f}")
        print(f"  非连通级联平均CC数: {np.mean([stats_num_components[i] for i in disconnected_idx]):.2f}")

    # 返回用于跨数据集汇总表的关键指标
    connected_ratio = len(connected_idx) / total * 100 if total > 0 else None
    mean_num_components = float(np.mean(stats_num_components)) if total > 0 else None
    orphan_cascade_ratio = has_orphan / total * 100 if total > 0 else None
    orphan_node_ratio_pct = float(np.mean(orphan_node_ratios) * 100) if orphan_node_ratios else None
    unified_ratio = len(source_cc_unified_sizes) / total * 100 if total > 0 else None
    unified_cc_avg_size = float(np.mean(source_cc_unified_sizes)) if source_cc_unified_sizes else None
    unified_largest_ratio = (
        float(np.mean(unified_source_cc_is_largest) * 100)
        if unified_source_cc_is_largest else None
    )
    singleton_cc_ratio = (
        float(np.mean(np.array(component_sizes_all) == 1) * 100)
        if component_sizes_all else None
    )
    max_cc_avg_ratio = float(np.mean(stats_max_component_ratio) * 100) if total > 0 else None

    return {
        'connected_ratio': connected_ratio,
        'mean_num_components': mean_num_components,
        'orphan_cascade_ratio': orphan_cascade_ratio,
        'orphan_node_ratio_pct': orphan_node_ratio_pct,
        'unified_ratio': unified_ratio,
        'unified_cc_avg_size': unified_cc_avg_size,
        'unified_largest_ratio': unified_largest_ratio,
        'singleton_cc_ratio': singleton_cc_ratio,
        'max_cc_avg_ratio': max_cc_avg_ratio,
    }


if __name__ == "__main__":
    datasets = ['douban_25c.SG', 'android_25c.SG', 'christianity_25c.SG', 'twitter_25c.SG']
    ds_display = {
        'christianity_25c.SG': 'Christianity',
        'douban_25c.SG': 'Douban',
        'android_25c.SG': 'Android',
        'twitter_25c.SG': 'Twitter',
    }
    table_order = ['christianity_25c.SG', 'douban_25c.SG', 'android_25c.SG', 'twitter_25c.SG']
    summary_by_dataset = {}

    for ds in datasets:
        print("\n" + "▓" * 70)
        print(f"  数据集: {ds}")
        print("▓" * 70 + "\n")
        try:
            summary_by_dataset[ds] = analyze_infected_subgraph(ds)
        except Exception as e:
            print(f"  错误: {e}")
        print()

    print("\n" + "=" * 70)
    print("# 感染子图连通性分析")
    print("孤岛 = 无源点的连通块")
    print("=" * 70)

    header = ["指标", "Christianity", "Douban", "Android", "Twitter"]
    print("\t".join(header))

    metric_defs = [
        ("感染子图完全连通", "connected_ratio", True),
        ("平均连通分量数", "mean_num_components", False),
        ("存在\"孤岛\"(无源点CC)的级联比例", "orphan_cascade_ratio", True),
        ("孤岛节点占感染节点的比例", "orphan_node_ratio_pct", True),
        ("源点 100% 聚集在同一 CC", "unified_ratio", True),
        ("100%聚集所在这个CC的平均大小", "unified_cc_avg_size", False),
        ("100%源点cc本级联最大CC的比例", "unified_largest_ratio", True),
        ("大小=1 的孤立节点占比(所有CC)", "singleton_cc_ratio", True),
        ("最大 CC 平均占比", "max_cc_avg_ratio", True),
    ]

    for label, key, is_percent in metric_defs:
        row = [label]
        for ds in table_order:
            stats = summary_by_dataset.get(ds)
            value = stats.get(key) if stats else None
            row.append(_fmt_table_value(value, is_percent=is_percent))
        print("\t".join(row))
