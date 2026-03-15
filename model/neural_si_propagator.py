"""
Neural SI Propagator (前向生成器)
可微神经 SI 传播器，用于从源点概率预测最终感染分布

设计原则:
    1. 全图结构 (Static Full Graph) + 掩码计算
    2. 不对称传染率预测器 (基于度数差异)
    3. 泊松-易感更新规则 (天然残差结构)
    4. K步传播 (K=2或3, 符合数据规律)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import degree


class AsymmetricInfectionRatePredictor(nn.Module):
    """
    不对称传染率预测器
    
    核心洞察: 源点具有更高的 Outward Ratio (向小度数节点传播)
    
    设计: 基于边两端度数差异的可学习传染概率
        p_{j→i} = Sigmoid(MLP([log(D_j), log(D_i)]))
    
    Args:
        hidden_dim: MLP 隐藏层维度 (默认 16)
    """
    def __init__(self, hidden_dim: int = 16):
        super().__init__()
        
        self.mlp = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.ReLU(),
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
    
    def forward(self, edge_index, degrees):
        """
        计算每条边的传染概率
        
        Args:
            edge_index: 边索引 [2, E]
            degrees: 节点度数 [N]
        
        Returns:
            infection_rates: 边传染概率 [E]
        """
        # 提取边两端节点的度数
        src_deg = degrees[edge_index[0]]  # [E]
        dst_deg = degrees[edge_index[1]]  # [E]
        
        # 对数变换 (稳定数值)
        log_src_deg = torch.log(src_deg + 1e-5)
        log_dst_deg = torch.log(dst_deg + 1e-5)
        
        # 拼接为特征对
        deg_pair = torch.stack([log_src_deg, log_dst_deg], dim=1)  # [E, 2]
        
        # MLP 预测传染概率
        infection_rates = self.mlp(deg_pair).squeeze(-1)  # [E]
        
        return infection_rates


class NeuralSIPropagator(nn.Module):
    """
    可微神经 SI 传播器
    
    核心流程:
        1. 初始化: H^(0) = P_S (源点概率)
        2. K步传播: H^(k+1) = H^(k) + (1-H^(k)) * (1 - exp(-β*R^(k)))
        3. 输出: Ô = H^(K) (最终感染分布)
    
    Args:
        k_steps: 传播步数 (默认 3)
        beta_init: 初始感染烈度 (默认 1.0)
        infection_rate_hidden: 传染率预测器隐藏层维度 (默认 16)
    """
    def __init__(
        self,
        k_steps: int = 3,
        beta_init: float = 1.0,
        infection_rate_hidden: int = 16
    ):
        super().__init__()
        
        self.k_steps = k_steps
        
        # 可学习的全局感染烈度 (控制传播速度)
        self.beta = nn.Parameter(torch.tensor(beta_init))
        
        # 不对称传染率预测器
        self.infection_rate_predictor = AsymmetricInfectionRatePredictor(
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
        
        # 预计算边传染率 (所有步共享)
        p_edge = self.infection_rate_predictor(edge_index, degrees)  # [E]
        
        # 初始化感染状态
        H = P_S.clone()  # [N]
        
        # K步传播
        for k in range(self.k_steps):
            H = self._poisson_update(H, edge_index, p_edge, num_nodes)
        
        return H
    
    def _poisson_update(self, H_curr, edge_index, p_edge, num_nodes):
        """
        泊松-易感更新规则 (天然残差结构)
        
        公式:
            R_i = Σ_{j∈N(i)} p_{j→i} * H_j  (邻居传染风险)
            H_i^(k+1) = H_i^(k) + (1 - H_i^(k)) * (1 - exp(-β*R_i))
        
        Args:
            H_curr: 当前感染状态 [N]
            edge_index: 边索引 [2, E]
            p_edge: 边传染率 [E]
            num_nodes: 节点总数
        
        Returns:
            H_next: 下一步感染状态 [N]
        """
        device = H_curr.device
        
        # 计算邻居传染风险
        src, dst = edge_index[0], edge_index[1]
        
        # R_i = Σ_{j∈N(i)} p_{j→i} * H_j
        weighted_infection = p_edge * H_curr[src]  # [E]
        
        # 聚合到目标节点
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
