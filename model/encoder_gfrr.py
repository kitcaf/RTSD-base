"""
GFRR Encoder 模块
基于 GAT 的共享编码器，将观测态/源点态特征映射到潜在空间

设计原则:
    1. 双通道输入: StateEmbedding + TopoProjection
    2. SE-Block 特征重校准
    3. 原始 GATv2 残差块提取图结构特征
    4. GatedFusion 多尺度融合
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import softmax, add_self_loops, remove_self_loops


# ==========================================
# 1. GATv2 Layer
# ==========================================
class GATv2Layer(MessagePassing):
    """
    原始 GATv2 风格注意力层（不使用感染密度梯度偏置）。
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        heads: int = 4,
        concat: bool = True,
        negative_slope: float = 0.2,
        dropout: float = 0.1,
        add_self_loops: bool = True,
        bias: bool = True,
        **kwargs
    ):
        kwargs.setdefault('aggr', 'add')
        super().__init__(node_dim=0, **kwargs)
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout
        self._add_self_loops = add_self_loops
        
        # 线性变换
        self.lin_l = nn.Linear(in_channels, heads * out_channels, bias=False)
        self.lin_r = nn.Linear(in_channels, heads * out_channels, bias=False)
        
        # 注意力参数
        self.att = Parameter(torch.Tensor(1, heads, out_channels))
        
        # 偏置
        if bias and concat:
            self.bias = Parameter(torch.Tensor(heads * out_channels))
        elif bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        
        self._reset_parameters()
    
    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.lin_l.weight)
        nn.init.xavier_uniform_(self.lin_r.weight)
        nn.init.xavier_uniform_(self.att)
        if self.bias is not None:
            nn.init.zeros_(self.bias)
    
    def forward(self, x, edge_index):
        """
        Args:
            x: 节点特征 [N, C]
            edge_index: 边索引 [2, E]
        """
        N = x.size(0)
        
        if self._add_self_loops:
            edge_index, _ = remove_self_loops(edge_index)
            edge_index, _ = add_self_loops(edge_index, num_nodes=N)
        
        x_l = self.lin_l(x).view(-1, self.heads, self.out_channels)
        x_r = self.lin_r(x).view(-1, self.heads, self.out_channels)
        
        out = self.propagate(edge_index, x=(x_l, x_r), size=None)
        
        if self.concat:
            out = out.view(-1, self.heads * self.out_channels)
        else:
            out = out.mean(dim=1)
        
        if self.bias is not None:
            out = out + self.bias
        
        return out
    
    def message(self, x_i, x_j, index, ptr, size_i):
        # GATv2 注意力
        x_sum = x_i + x_j
        x_sum = F.leaky_relu(x_sum, negative_slope=self.negative_slope)
        alpha = (x_sum * self.att).sum(dim=-1)  # [E, heads]
        
        alpha = softmax(alpha, index, ptr, size_i)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        return alpha.unsqueeze(-1) * x_j


# ==========================================
# 2. 双通道输入层
# ==========================================
class DualChannelInput(nn.Module):
    """
    双通道输入层: 状态嵌入 + 拓扑投影
    
    设计原则:
        - StateEmbedding: 将 0/1 状态映射为语义向量
        - TopoProjection: 将 13 维拓扑特征投影到隐藏空间
        - SE-Block: 自适应特征重校准
    
    Args:
        state_dim: 状态嵌入维度 (默认 8)
        topo_dim: 拓扑投影维度 (默认 24)
        num_topo_features: 拓扑特征数量 (默认 13)
    """
    def __init__(self, state_dim=8, topo_dim=24, num_topo_features=13):
        super().__init__()
        
        self.state_dim = state_dim
        self.topo_dim = topo_dim
        self.out_dim = state_dim + topo_dim
        
        # 状态嵌入: 0 -> "未感染/非源点", 1 -> "感染/源点"
        self.state_embedding = nn.Embedding(2, state_dim)
        
        # 拓扑投影
        self.topo_projection = nn.Sequential(
            nn.Linear(num_topo_features, topo_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(topo_dim, topo_dim)
        )
        
        # SE-Block 特征重校准
        self.se_block = nn.Sequential(
            nn.Linear(self.out_dim, self.out_dim // 2),
            nn.ReLU(),
            nn.Linear(self.out_dim // 2, self.out_dim),
            nn.Sigmoid()
        )
        
        self._reset_parameters()
    
    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.state_embedding.weight)
        for layer in self.topo_projection:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)
    
    def forward(self, x):
        """
        Args:
            x: [N, D] 特征矩阵, 第0列是状态标志
        
        Returns:
            h_fused: [N, state_dim + topo_dim] 融合特征
        """
        # 分离状态和拓扑特征
        state_col = x[:, 0].long().clamp(0, 1)  # [N]
        topo_cols = x[:, 1:]  # [N, 13]
        
        # 双通道映射
        h_state = self.state_embedding(state_col)  # [N, state_dim]
        h_topo = self.topo_projection(topo_cols)   # [N, topo_dim]
        
        # 拼接
        h_cat = torch.cat([h_state, h_topo], dim=-1)  # [N, out_dim]
        
        # SE 重校准
        weights = self.se_block(h_cat)
        h_fused = h_cat * weights
        
        return h_fused


# ==========================================
# 3. GATv2 残差块
# ==========================================
class PIRAResidualBlock(nn.Module):
    """
    GATv2 残差块
    
    架构: GATv2Layer -> LeakyReLU -> Dropout + Skip Connection
    """
    def __init__(self, channels, heads=4, dropout=0.3):
        super().__init__()
        
        out_per_head = channels // heads
        self.gat = GATv2Layer(
            in_channels=channels,
            out_channels=out_per_head,
            heads=heads,
            concat=True,
            dropout=0.1
        )
        self.activation = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, edge_index):
        residual = self.gat(x, edge_index)
        residual = self.activation(residual)
        out = x + residual
        out = self.dropout(out)
        return out


# ==========================================
# 4. 门控融合模块
# ==========================================
class GatedFusion(nn.Module):
    """
    自适应门控融合: 融合多层 GAT 输出
    """
    def __init__(self, in_channels):
        super().__init__()
        
        self.gate_net = nn.Sequential(
            nn.Linear(in_channels * 2, in_channels // 2),
            nn.ReLU(),
            nn.Linear(in_channels // 2, 1),
            nn.Sigmoid()
        )
    
    def forward(self, h1, h2):
        """
        Args:
            h1: 浅层特征 [N, C]
            h2: 深层特征 [N, C]
        
        Returns:
            h_out: 融合特征 [N, C]
            gate: 门控权重 [N, 1]
        """
        cat_feat = torch.cat([h1, h2], dim=-1)
        gate = self.gate_net(cat_feat)
        h_out = gate * h1 + (1 - gate) * h2
        return h_out, gate


# ==========================================
# 5. 最大CC Pool + 回注
# ==========================================
class MaxCCPoolInjector(nn.Module):
    """
    仅在最大CC上做 pool 并回注到节点表示。

    设计:
        1. 取最大CC节点集合 Cmax
        2. z_cc_max = mean(h[Cmax])
        3. 可选 MLP 变换 z_cc_max
        4. 最大CC内强回注, 外部可选弱回注
    """
    def __init__(self, hidden_dim: int, use_mlp: bool = True):
        super().__init__()
        self.use_mlp = use_mlp
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim)
        ) if use_mlp else nn.Identity()

    def forward(self, h, max_cc_mask, alpha: float = 0.2, outside_alpha: float = 0.0):
        """
        Args:
            h: [N, C] 节点表示
            max_cc_mask: [N] bool, 最大CC掩码
            alpha: 最大CC内回注强度
            outside_alpha: 最大CC外回注强度（默认0）

        Returns:
            h_out: [N, C] 回注后表示
            z_cc_max: [C] 最大CC池化向量（用于可解释分析）
        """
        if max_cc_mask is None:
            return h, None

        mask = max_cc_mask.bool().to(h.device)
        if mask.sum() == 0:
            return h, None

        z_cc_max = h[mask].mean(dim=0, keepdim=True)  # [1, C]
        z_cc_max = self.mlp(z_cc_max)

        h_out = h.clone()
        h_out[mask] = h_out[mask] + alpha * z_cc_max
        if outside_alpha > 0:
            outside_mask = ~mask
            if outside_mask.sum() > 0:
                h_out[outside_mask] = h_out[outside_mask] + outside_alpha * z_cc_max

        return h_out, z_cc_max.squeeze(0)


# ==========================================
# 6. GFRR Encoder 主模型
# ==========================================
class GFRREncoder(nn.Module):
    """
    GFRR 编码器: 将输入特征映射到潜在空间
    
    架构:
        Input [N, D]
            ↓
        DualChannelInput (State + Topo)
            ↓
        GATv2 Residual Blocks × num_blocks
            ↓
        GatedFusion (多尺度融合)
            ↓
        z [N, hidden_dim] (潜在表示)
    
    Args:
        num_features: 输入特征维度 (默认 14)
        hidden_dim: 隐藏层维度 (默认 32)
        num_blocks: GAT 残差块数量 (默认 3)
        dropout: Dropout 比例 (默认 0.3)
    """
    def __init__(
        self,
        num_features: int = 22,
        hidden_dim: int = 32,
        num_blocks: int = 3,
        dropout: float = 0.3,
        use_max_cc_pool: bool = True,
        max_cc_pool_use_mlp: bool = True,
        max_cc_pool_alpha: float = 0.2,
        max_cc_pool_outside_alpha: float = 0.0
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_blocks = num_blocks
        
        # 动态计算双通道维度 (总和为 hidden_dim)
        state_dim = max(4, hidden_dim // 4)
        topo_dim = hidden_dim - state_dim
        
        # 双通道输入
        self.input_layer = DualChannelInput(
            state_dim=state_dim,
            topo_dim=topo_dim,
            num_topo_features=num_features - 1  # 除去状态列
        )
        
        # GATv2 残差块
        self.res_blocks = nn.ModuleList([
            PIRAResidualBlock(
                hidden_dim, 
                heads=4, 
                dropout=dropout
            )
            for _ in range(num_blocks)
        ])
        
        # 门控融合
        self.gated_fusion = GatedFusion(hidden_dim)

        # 最大CC Pool + 回注
        self.use_max_cc_pool = use_max_cc_pool
        self.max_cc_pool_alpha = max_cc_pool_alpha
        self.max_cc_pool_outside_alpha = max_cc_pool_outside_alpha
        self.max_cc_pool = MaxCCPoolInjector(
            hidden_dim=hidden_dim,
            use_mlp=max_cc_pool_use_mlp
        )
        self.last_z_cc_max = None
    
    def forward(self, x, edge_index, max_cc_mask=None):
        """
        前向传播
        
        Args:
            x: 节点特征 [N, D]
            edge_index: 边索引 [2, E]
            
        Returns:
            z: 潜在表示 [N, hidden_dim]
            gate_weights: 门控权重 [N, 1] (可选, 用于分析)
        """
        # 双通道输入
        h = self.input_layer(x)  # [N, 32]
        
        # GATv2 残差块
        layer_outputs = []
        for block in self.res_blocks:
            h = block(h, edge_index)
            layer_outputs.append(h)

        # 门控融合 (融合第一层和最后一层)
        z, gate_weights = self.gated_fusion(layer_outputs[0], layer_outputs[-1])

        # 最大CC Pool + 残差回注
        if self.use_max_cc_pool:
            z, z_cc_max = self.max_cc_pool(
                z,
                max_cc_mask=max_cc_mask,
                alpha=self.max_cc_pool_alpha,
                outside_alpha=self.max_cc_pool_outside_alpha
            )
            self.last_z_cc_max = z_cc_max
        else:
            self.last_z_cc_max = None

        return z, gate_weights

    def encode(self, x, edge_index, max_cc_mask=None):
        """简化接口: 仅返回潜在表示"""
        z, _ = self.forward(x, edge_index, max_cc_mask=max_cc_mask)
        return z
