"""
Neural SI Propagator (前向生成器)
可微神经 SI 传播器，用于从源点概率预测最终感染分布

设计原则:
    1. 全图结构 (Static Full Graph) + 掩码计算
    2. 动态传染率预测器 (基于节点特征 + 当前感染状态 + 度数)
    3. 泊松-易感更新规则 (天然残差结构)
    4. K步传播 (K=2或3, 符合数据规律)
    
改进点:
    - 传染率不再是静态的，而是每步动态计算
    - 利用源点概率 P_S 作为节点语义特征
    - 利用当前感染状态 H^(k) 作为动态信号
    - 结合度数信息捕捉拓扑不对称性
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import degree
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import softmax


class DynamicInfectionRatePredictor(nn.Module):
    """
    动态传染率预测器 (改进版)
    
    核心改进: 传染率不仅基于度数，还基于节点的源点概率和当前感染状态
    
    公式:
        p_{j→i}^(k) = Sigmoid(MLP([P_S_j, P_S_i, H_j^(k), H_i^(k), log(D_j), log(D_i)]))
    
    直觉:
        - P_S_j 高 → 源点 → 传染力强
        - H_j^(k) 高 → 已感染 → 传染力强
        - D_j > D_i → 大V传小V → 传染力强 (Outward Ratio)
    
    Args:
        hidden_dim: MLP 隐藏层维度 (默认 32)
    """
    def __init__(self, hidden_dim: int = 32):
        super().__init__()
        
        # 输入: [P_S_j, P_S_i, H_j, H_i, log(D_j), log(D_i)] = 6 维
        self.mlp = nn.Sequential(
            nn.Linear(6, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
        self._reset_parameters()
    
    def _reset_parameters(self):
        for layer in self.mlp:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)
    
    def forward(self, edge_index, P_S, H_curr, degrees):
        """
        动态计算每条边的传染概率
        
        Args:
            edge_index: 边索引 [2, E]
            P_S: 源点概率 [N] (来自后向判别器, 静态)
            H_curr: 当前感染状态 [N] (动态)
            degrees: 节点度数 [N]
        
        Returns:
            infection_rates: 边传染概率 [E]
        """
        src, dst = edge_index[0], edge_index[1]
        
        # 提取边两端的特征
        P_S_src = P_S[src]  # [E]
        P_S_dst = P_S[dst]  # [E]
        H_src = H_curr[src]  # [E]
        H_dst = H_curr[dst]  # [E]
        
        # 度数特征
        log_deg_src = torch.log(degrees[src] + 1e-5)
        log_deg_dst = torch.log(degrees[dst] + 1e-5)
        
        # 拼接为边特征 [E, 6]
        edge_features = torch.stack([
            P_S_src, P_S_dst,
            H_src, H_dst,
            log_deg_src, log_deg_dst
        ], dim=1)
        
        # MLP 预测传染概率
        infection_rates = self.mlp(edge_features).squeeze(-1)  # [E]
        
        return infection_rates


class NeuralSIPropagator(nn.Module):
    """
    可微神经 SI 传播器 (改进版)
    
    核心流程:
        1. 初始化: H^(0) = P_S (源点概率)
        2. K步动态传播:
           - 计算动态传染率: p_{j→i}^(k) = f(P_S, H^(k), degrees)
           - 泊松更新: H^(k+1) = H^(k) + (1-H^(k)) * (1 - exp(-β*R^(k)))
        3. 输出: Ô = H^(K) (最终感染分布)
    
    改进点:
        - 传染率每步动态计算 (感知当前感染状态)
        - 利用源点概率 P_S 作为节点语义
        - 结合度数捕捉拓扑不对称性
    
    Args:
        k_steps: 传播步数 (默认 3)
        beta_init: 初始感染烈度 (默认 1.0)
        infection_rate_hidden: 传染率预测器隐藏层维度 (默认 32)
    """
    def __init__(
        self,
        k_steps: int = 3,
        beta_init: float = 1.0,
        infection_rate_hidden: int = 32
    ):
        super().__init__()
        
        self.k_steps = k_steps
        
        # 可学习的全局感染烈度 (控制传播速度)
        self.beta = nn.Parameter(torch.tensor(beta_init))
        
        # 动态传染率预测器
        self.infection_rate_predictor = DynamicInfectionRatePredictor(
            hidden_dim=infection_rate_hidden
        )
    
    def forward(self, P_S, edge_index, degrees, num_nodes=None):
        """
        前向传播: 从源点概率预测感染分布
        
        Args:
            P_S: 源点概率 [N] (来自后向判别器)
            edge_index: 边索引 [2, E]
            degrees: 节点度数 [N]
            num_nodes: 节点总数 (可选, 用于验证)
        
        Returns:
            O_pred: 预测的感染分布 [N]
        """
        if num_nodes is None:
            num_nodes = P_S.size(0)
        
        device = P_S.device
        
        # 初始化感染状态
        H = P_S.clone()  # [N]
        
        # K步动态传播
        for k in range(self.k_steps):
            # 动态计算传染率 (每步都重新计算)
            p_edge = self.infection_rate_predictor(edge_index, P_S, H, degrees)
            
            # 泊松更新
            H = self._poisson_update(H, edge_index, p_edge, num_nodes)
        
        return H
    
    def _poisson_update(self, H_curr, edge_index, p_edge, num_nodes):
        """
        泊松-易感更新规则 (天然残差结构)
        
        公式:
            R_i = Σ_{j∈N(i)} p_{j→i}^(k) * H_j^(k)  (邻居传染风险)
            H_i^(k+1) = H_i^(k) + (1 - H_i^(k)) * (1 - exp(-β*R_i))
        
        Args:
            H_curr: 当前感染状态 [N]
            edge_index: 边索引 [2, E]
            p_edge: 边传染率 [E] (动态计算)
            num_nodes: 节点总数
        
        Returns:
            H_next: 下一步感染状态 [N]
        """
        device = H_curr.device
        
        # 计算邻居传染风险
        src, dst = edge_index[0], edge_index[1]
        
        # R_i = Σ_{j∈N(i)} p_{j→i} * H_j
        weighted_infection = p_edge * H_curr[src]  # [E]
        
        # 聚合到目标节点 (这里用了图结构!)
        R = torch.zeros(num_nodes, device=device)
        R.index_add_(0, dst, weighted_infection)  # [N]
        
        # 泊松感染概率: 1 - exp(-β*R)
        # 数值稳定性: clamp β*R 避免 exp 下溢
        infection_prob = 1.0 - torch.exp(-torch.clamp(self.beta * R, max=10.0))
        
        # 易感余量
        susceptible = 1.0 - H_curr
        
        # 天然残差更新
        H_next = H_curr + susceptible * infection_prob
        
        # 确保 [0, 1] 范围
        H_next = torch.clamp(H_next, 0.0, 1.0)
        
        return H_next
    
    def get_beta(self):
        """获取当前感染烈度"""
        return self.beta.item()
