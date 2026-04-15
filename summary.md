# MaxCC 双口径分析总结

> 统计源点在最大感染联通图的性质分布等情况

## 分析对象

本总结基于以下 4 份最新报告：

- [android/report.txt](E:/desk/diffuse/diff_true/SII/RTSD-v1/analysis_outputs/maxcc_pure_input/android/report.txt)
- [christianity/report.txt](E:/desk/diffuse/diff_true/SII/RTSD-v1/analysis_outputs/maxcc_pure_input/christianity/report.txt)
- [douban/report.txt](E:/desk/diffuse/diff_true/SII/RTSD-v1/analysis_outputs/maxcc_pure_input/douban/report.txt)
- [twitter/report.txt](E:/desk/diffuse/diff_true/SII/RTSD-v1/analysis_outputs/maxcc_pure_input/twitter/report.txt)

这里的源点形态分布同时给出两种口径：

- `2-hop 邻近`：两个源点在 MaxCC 内距离 `<=2` 就算同一局部块
- `1-hop 邻近`：两个源点在 MaxCC 内必须直接相连才算同一局部块

源点形态分成 5 类：

- `singleton`
- `簇`
- `多区块`
- `链`
- `松散团`

其中 `松散团` 是独立兜底类。即使某类样本数为 `0`，报告里也会显式展示该类。

## 核心结论

### 1. MaxCC 仍然是有效的主搜索空间，但不同数据集差异很大

四个数据集的 `all_sources_in_maxcc_ratio` 为：

| 数据集 | all_sources_in_maxcc_ratio | 结论 |
| --- | --- | --- |
| `christianity` | `0.9927` | 几乎所有源点都在 MaxCC 内，MaxCC 基本就是主要搜索空间 |
| `douban` | `0.9171` | MaxCC 很有用，大多数源点都能覆盖到 |
| `android` | `0.8887` | MaxCC 仍然有效，但已经存在一定漏检风险 |
| `twitter` | `0.7366` | MaxCC 只能作为主候选区，不能作为唯一候选区 |

可以直接得到：

- `christianity / douban / android` 上，先在 MaxCC 内做识别是合理的。
- `twitter` 上，不能只依赖 MaxCC，MaxCC 外还需要保留补充候选。

### 2. 在 2-hop 口径下，源点往往呈局部簇状；在 1-hop 口径下，会明显裂解成更多多区块

这是这次分析最重要的形态结论。

四个数据集都表现出一致趋势：

- `2-hop` 下，`簇` 的比例更高
- `1-hop` 下，很多原本的 `簇` 会转成 `多区块`
- `松散团` 在 `2-hop` 下四个数据集都是 `0.0000`
- `松散团` 在 `1-hop` 下只占很小比例，说明它不是主导模式，只是少量残余结构

这说明：

- 源点通常彼此接近
- 但这种接近更多是 `2-hop` 局部接近，而不一定是 `1-hop` 直接相连

更准确地说：

- 源点往往属于同一个局部传播区域
- 但不一定在社交图上直接连边形成一个紧密团

### 3. 源点不像“深层图核心”，更像“局部强节点”

四个数据集的 `source_boundary_depth_mean` 都非常接近 `0`，说明源点并不表现为“位于 MaxCC 深层内部”的节点。

结合特征排序，源点更像：

- 在局部感染结构里连接更多感染节点
- 相对邻居更强
- 在局部上更居中一些

而不是：

- 位于最大 `k-core`
- 离感染边界很远

因此，“源点是深层核心节点”这个解释并不成立。

## 2-hop 与 1-hop 的具体对比

### Android

- `2-hop`：`簇 0.6186`，`singleton 0.2392`，`多区块 0.0763`，`链 0.0660`，`松散团 0.0000`
- `1-hop`：`多区块 0.4660`，`簇 0.2619`，`singleton 0.2392`，`链 0.0206`，`松散团 0.0124`

解释：

- `2-hop` 下 Android 主要表现为局部簇
- 一旦改成更严格的 `1-hop`，最大类立刻变成 `多区块`
- 说明很多源点并不直接连边，但它们通常处在彼此很近的局部传播区

### Christianity

- `2-hop`：`簇 0.8636`，其余四类都很少，`松散团 0.0000`
- `1-hop`：`簇 0.7018`，`多区块 0.1836`，`singleton 0.1018`，`链 0.0091`，`松散团 0.0036`

解释：

- 即使改成 `1-hop`，`簇` 仍然压倒性占优
- 说明这个数据集的源点确实更紧密、更局部、更直接相连

这是四个数据集中“紧密簇结构”最强的一个。

### Douban

- `2-hop`：`singleton 0.4444`，`簇 0.4305`，`链 0.0675`，`多区块 0.0576`，`松散团 0.0000`
- `1-hop`：`singleton 0.4444`，`簇 0.3078`，`多区块 0.2257`，`链 0.0201`，`松散团 0.0020`

解释：

- Douban 有大量 `singleton`
- 剩下的非单点源在 `2-hop` 下常表现为簇
- 但改成 `1-hop` 后，其中相当一部分会被拆成 `多区块`

因此 Douban 的形态特征是：

- 一部分级联在 MaxCC 内本来就只有 0 或 1 个源点
- 另一部分虽然局部接近，但未必直接相连

### Twitter

- `2-hop`：`簇 0.4143`，`singleton 0.4110`，`多区块 0.0924`，`链 0.0824`，`松散团 0.0000`
- `1-hop`：`singleton 0.4110`，`簇 0.2826`，`多区块 0.2404`，`链 0.0355`，`松散团 0.0305`

解释：

- Twitter 是四个数据集中最分散、也最难的一个
- MaxCC 覆盖最低
- 改成 `1-hop` 后，多区块继续显著增加

所以 Twitter 上最安全的判断是：

- 源点既不总在 MaxCC 内
- 即使落在 MaxCC 内，也更容易表现为碎片化局部块，而不是单一紧密簇

## 关于“松散团”的结论

`松散团` 现在作为独立兜底类存在，但从结果上看：

- 在 `2-hop` 口径下，四个数据集都是 `0.0000`
- 在 `1-hop` 口径下，只在 `android / christianity / douban / twitter` 上分别占 `0.0124 / 0.0036 / 0.0020 / 0.0305`

可以直接解释为：

- `2-hop` 口径足够宽松时，几乎不会留下“既不成簇、也不成链、也不分块”的残余样本
- `1-hop` 口径更严格时，会析出少量“单区块但既不密也不长”的松散结构

所以：

- `松散团` 是必要的语义兜底类
- 但它不是主导模式

## 在 MaxCC 内最有效的特征

四个报告的排序高度一致，最稳的结论没有因为加入 `1-hop / 2-hop` 双口径而改变。

### 第一梯队：最值得优先使用

| 特征 | 作用理解 | 结论 |
| --- | --- | --- |
| `outward_ratio_in_maxcc` | 在 MaxCC 邻域里，自己是否比邻居更强 | 跨数据集最稳定，应该作为核心打分项 |
| `infected_degree_in_maxcc` | 在 MaxCC 内连接了多少感染节点 | 对源点识别非常有效，尤其在 `douban / twitter` 上突出 |
| `global_degree` | 在原始底层图中的总连接能力 | 稳定有效，说明源点本身更像强连接节点 |

这三类特征共同说明：

- 源点更像“局部强节点”
- 而不是“深层核心节点”

### 第二梯队：强辅助特征

- `harmonic_in_maxcc`
- `closeness_in_maxcc`
- `avg_distance_in_maxcc`
- `stronger_neighbor_ratio`
- `kinf_vs_best_neighbor_ratio`

这些特征的价值在于：

- 它们补充了“局部中心性”和“相对邻居优势”
- 和第一梯队特征组合后更适合做联合打分

### `K-core` 的结论

`kcore_in_maxcc` 仍然不是稳定主特征：

- `christianity` 上表现较强
- `android / douban / twitter` 上都不突出

因此：

- 可以保留
- 但不应该把“高 k-core”作为源点判断的主要依据

## 明确无效或偏弱的结论

以下判断在这次分析里都不成立，或者只具备较弱解释力：

1. “源点通常位于 MaxCC 深层内部”
2. “源点一定是直接连边形成的紧密团”
3. “K-core 是最核心的源点特征”
4. “只看 MaxCC 就足够覆盖所有源点”

对应原因是：

- `boundary_depth_in_maxcc` 很弱
- `1-hop` 下大量 `簇` 会转成 `多区块`
- `kcore_in_maxcc` 只在个别数据集上强
- `twitter` 的 MaxCC 覆盖明显不足

## 分层结果能说明什么

`[难度分层]` 这部分最重要的作用，不是单看某个桶的均值，而是帮助我们判断：

- `MaxCC` 在什么条件下是可靠的主搜索空间
- 问题什么时候会变难
- 难是因为源点没进 `MaxCC`，还是进了 `MaxCC` 但过于分散

### 1. `source_spread` 是最关键的难度分层变量

这是最强的结构性信号。

- 当 `source_spread = 1` 时，源点都在同一个感染连通块里，`MaxCC` 最有机会覆盖全部源点
- 在 `android / christianity / douban / twitter` 上，这一桶的 `all_in_maxcc` 约为 `0.966 / 0.993 / 0.968 / 0.925`
- 一旦 `source_spread > 1`，`all_in_maxcc` 基本就掉到 `0`

结论：

- `source_spread` 决定了 `MaxCC` 方法有没有结构性上限
- 如果源点天然落在多个感染块里，单靠 `MaxCC` 不可能把所有源点都找全

### 2. `source_density` 越低，问题通常越难

低密度表示：

- `MaxCC` 很大
- 真实源点在其中很稀

这会同时带来两个困难：

- 候选空间更大
- 源点不够集中

最明显的是 `twitter`：

- 在 `source_density <= 0.10` 这一桶里，`all_in_maxcc` 只有 `0.425`

`android` 和 `douban` 的低密度桶也明显变差，约都在 `0.77` 左右；`christianity` 对低密度更不敏感，因为它整体结构本来就更集中。

结论：

- 源点越稀，`MaxCC` 内识别越难
- 这个效应在 `twitter` 上最明显

### 3. `source_count` 和 `max_cc_size` 会增加难度，但它们更多是间接变量

随着源点数变多、`MaxCC` 变大，通常会看到：

- `pair_dist` 上升
- 源点密度下降
- 源点更容易分散

例如 `twitter`：

- `source_count = 1` 时，`all_in_maxcc ≈ 0.868`
- `source_count = 7+` 时，降到 `0.550`

但这个趋势并不是所有数据集都一样强，例如 `christianity` 即使 `MaxCC` 很大，覆盖率依然接近 `1`。

结论：

- `source_count` 和 `max_cc_size` 不是最本质的因子
- 它们主要通过“让源点更分散”来提高难度

### 4. 分层结果再次证明：源点不像深层核心，更像局部强节点

无论在哪个分层桶里，`boundary_depth` 都普遍接近 `0`。

这说明：

- 源点并不会因为传播规模变大、源点数变多，就变成 `MaxCC` 深处的节点
- 相反，源点始终更像靠近感染边界、但在局部更强的一批点

结论：

- `boundary_depth_in_maxcc` 不是稳定强特征
- 分层结果强化了“源点是局部强节点，而不是深层核心”的总体判断

### 5. 最容易和最困难的场景

最容易的场景：

- `source_spread = 1`
- `source_density` 高
- `christianity` 整体
- 小到中等规模的 `MaxCC`

最困难的场景：

- `source_spread > 1`
- `source_density <= 0.10`
- `twitter`
- 大 `MaxCC` 且多源点的级联

## 对后续建模最有用的建议

### 候选空间

- 对 `christianity / douban / android`，优先在 MaxCC 内做识别是合理的
- 对 `twitter`，应该保留 MaxCC 外补充候选

### 特征设计

优先保留：

- `outward_ratio_in_maxcc`
- `infected_degree_in_maxcc`
- `global_degree`
- `harmonic_in_maxcc`
- `closeness_in_maxcc`
- `stronger_neighbor_ratio`
- `kinf_vs_best_neighbor_ratio`

可以弱化：

- `internal_degree_ratio`
- `boundary_depth_in_maxcc`

### 形态先验

如果后续还要把“源点形态”当成分析先验，建议同时保留两种口径：

- `2-hop`：更适合表达“是否属于同一局部传播区”
- `1-hop`：更适合表达“是否直接相连形成紧密簇”

这比只说“源点是簇状”更准确。

## 最终结论

把这次最新分析压成最关键的 7 句话，就是：

1. MaxCC 对源点识别总体是有价值的，但 `twitter` 不能只依赖 MaxCC。
2. 源点并不表现为 MaxCC 内部的深层核心，而更像局部传播中的强节点。
3. 在 `2-hop` 口径下，源点经常表现为局部簇，而且四个数据集都没有 `松散团` 残余类。
4. 但在 `1-hop` 口径下，大量“簇”会变成“多区块”，说明很多源点只是局部接近，不一定直接连边。
5. `松散团` 在 `1-hop` 下只占很小比例，因此它是必要的兜底类，但不是主导模式。
6. 最稳定有效的识别特征是 `outward_ratio_in_maxcc`、`infected_degree_in_maxcc`、`global_degree`。
7. 后续更合适的方向是“MaxCC 内多特征联合打分 + 必要时保留 MaxCC 外补充候选”，而不是继续寻找单一决定性特征。
