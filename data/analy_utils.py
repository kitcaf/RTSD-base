def analyze_social_explainability(adj_sets, unique_users_ordered, source_indices):
    """
    计算单个级联的社交网络解释力指标。
    
    参数:
        adj_sets: dict, 邻接表 {user_idx: set(neighbor_indices)}
        unique_users_ordered: list, 按时间排序的去重用户ID列表
        source_indices: set, 源节点ID集合（注意：是节点ID，不是索引）
        
    返回:
        (targets, hop1, hop2, hop3plus, hop1_from_source, hop2_depend_source): tuple
        
    定义：
        d(u) = 对于非源节点u，在社交图中到所有前驱节点（比它早感染的人）的最短路径长度
        Hop-1: d(u) = 1，u 与某个前驱节点直接相连
        Hop-2: d(u) = 2，u 与某个前驱节点距离为2
        Hop-3+: d(u) >= 3，u 与所有前驱节点距离都 >= 3
    """
    # 状态池
    infected_pool = set()             # 已感染节点集合 (Potential Predecessors)
    infected_neighbors_pool = set()   # 已感染节点的邻居集合 (用于判断 Hop-2)
    
    cnt_targets = 0
    cnt_hop1 = 0
    cnt_hop2 = 0
    cnt_hop3plus = 0
    # Hop-1 中，前驱节点是源点的数量
    cnt_hop1_from_source = 0
    # Hop-2 中，依赖的感染节点是源点的数量 (u-w-v 中 v 是源点)
    cnt_hop2_depend_source = 0
    
    for i, u in enumerate(unique_users_ordered):
        # 1. 如果是源点：直接入池，不统计
        if u in source_indices:
            infected_pool.add(u)
            infected_neighbors_pool.update(adj_sets[u])
            continue
        
        # 2. 如果是目标节点：判断它离 infected_pool 有多远
        cnt_targets += 1
        u_neighbors = adj_sets[u]

        # Check Hop-1: u 的邻居里有已感染节点吗
        hop1_predecessors = u_neighbors & infected_pool
        if hop1_predecessors:
            cnt_hop1 += 1
            # 检查这些前驱节点中是否有源点
            if not hop1_predecessors.isdisjoint(source_indices):
                cnt_hop1_from_source += 1
            
        # Check Hop-2: u-w-v 路径，w 是 u 的邻居，v 是已感染节点
        # u 的邻居集合与 infected_neighbors_pool 有交集，说明存在 Hop-2 路径
        elif not u_neighbors.isdisjoint(infected_neighbors_pool):
            cnt_hop2 += 1
            # 找到 u-w-v 路径中的 v（已感染节点），检查 v 是否是源点
            # w 是 u 的邻居，v 是 w 的邻居且在 infected_pool 中
            found_source_dependency = False
            for w in u_neighbors:
                if w in infected_neighbors_pool:  # w 是某个已感染节点的邻居
                    # 找 w 的邻居中哪些是已感染的（这些就是 v）
                    w_neighbors = adj_sets[w]
                    v_candidates = w_neighbors & infected_pool
                    if not v_candidates.isdisjoint(source_indices):
                        found_source_dependency = True
                        break
            if found_source_dependency:
                cnt_hop2_depend_source += 1
        
        # Hop >= 3: 距离太远，无法追溯依赖路径
        else:
            cnt_hop3plus += 1
        
        # 3. 统计完后，将自己加入池子，作为后续节点的潜在前驱
        infected_pool.add(u)
        infected_neighbors_pool.update(adj_sets[u])
        
    return (cnt_targets, cnt_hop1, cnt_hop2, cnt_hop3plus, 
            cnt_hop1_from_source, cnt_hop2_depend_source)