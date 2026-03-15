### 数据集描述文档：传播溯源数据集 (.SG 格式)

> 包括：android、christianity、douban、twitter

#### 1\. 基本信息 (Basic Information)

  * **原始数据源**：Twitter 社交网络数据（包含 `edges.txt` 和 `cascades.txt`）。
  * **文件格式**：Python Pickle (`.SG` 文件)，包含自定义 `SparseGraph` 对象。
  * **适用任务**：信息传播源定位 (Information Source Tracing/Localization)、传播级联覆盖范围预测。
  * **图拓扑类型**：无向图 (Undirected Graph)。

#### 2\. 数据结构 (Data Structure)

该数据集被封装为一个 `SparseGraph` 对象，主要包含以下两个核心属性：

**A. 网络邻接矩阵 (`adj_matrix`)**

  * **类型**：`scipy.sparse.csr_matrix` (稀疏矩阵)
  * **形状**：$(N, N)$，其中 $N$ 为唯一用户节点总数。
  * **数值**：二值矩阵（0或1）。$A_{ij}=1$ 表示用户 $i$ 和用户 $j$ 之间存在社交关系。
  * **构建逻辑**：基于原始边列表构建。脚本配置为 `DIRECTED_GRAPH = False`，因此矩阵是对称的（即视为无向社交网络）。

**B. 传播级联张量 (`influ_mat_list`)**

  * **类型**：`numpy.ndarray` (稠密数组，但通常由稀疏数据转换而来)
  * **形状**：$(M, N, T)$
      * $M$：样本数（有效级联的总数量）。
      * $N$：节点总数（与邻接矩阵一致）。
      * $T$：时间快照数，固定为 **2** (`NUM_TIMESTEPS = 2`)。
  * **维度含义**：
      * **$T=0$ (Source/Seed Channel)**：表示传播的**源头状态**。
          * 数值为 1.0 表示该节点是源节点，0 表示非源。
      * **$T=1$ (Final/Activated Channel)**：表示传播的**最终状态**。
          * 数值为 1.0 表示该节点在级联结束时被感染/激活，0 表示未感染。
