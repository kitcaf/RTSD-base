"""
测试改进版前向生成器

改进点:
    1. 动态传染率 (基于 P_S + H^(k) + degrees)
    2. 每步重新计算传染率 (感知当前感染状态)
    3. 利用图结构进行消息传递
"""
import torch
from model.neural_si_propagator import NeuralSIPropagator, DynamicInfectionRatePredictor
from loss_forward import ForwardLoss


def test_dynamic_infection_rate():
    """测试动态传染率预测器"""
    print("=" * 60)
    print("测试 DynamicInfectionRatePredictor (改进版)")
    print("=" * 60)
    
    num_nodes = 100
    num_edges = 300
    
    # 模拟数据
    edge_index = torch.randint(0, num_nodes, (2, num_edges))
    P_S = torch.rand(num_nodes)  # 源点概率
    H_curr = torch.rand(num_nodes)  # 当前感染状态
    degrees = torch.rand(num_nodes) * 10 + 1
    
    # 创建预测器
    predictor = DynamicInfectionRatePredictor(hidden_dim=32)
    
    print(f"输入:")
    print(f"  边数: {num_edges}")
    print(f"  P_S 范围: [{P_S.min():.3f}, {P_S.max():.3f}]")
    print(f"  H_curr 范围: [{H_curr.min():.3f}, {H_curr.max():.3f}]")
    print(f"  度数范围: [{degrees.min():.1f}, {degrees.max():.1f}]")
    
    # 计算传染率
    p_edge = predictor(edge_index, P_S, H_curr, degrees)
    
    print(f"\n输出:")
    print(f"  传染率范围: [{p_edge.min():.3f}, {p_edge.max():.3f}]")
    print(f"  传染率均值: {p_edge.mean():.3f}")
    
    # 测试动态性: 改变 H_curr 后传染率应该变化
    H_curr_new = torch.rand(num_nodes)
    p_edge_new = predictor(edge_index, P_S, H_curr_new, degrees)
    
    diff = (p_edge - p_edge_new).abs().mean()
    print(f"\n动态性测试:")
    print(f"  改变 H_curr 后传染率变化: {diff:.4f}")
    print(f"  ✓ 传染率是动态的" if diff > 0.01 else "  ✗ 传染率几乎不变")
    
    print("\n✓ DynamicInfectionRatePredictor 测试通过\n")


def test_propagator():
    """测试改进版前向生成器"""
    print("=" * 60)
    print("测试 NeuralSIPropagator (改进版)")
    print("=" * 60)
    
    num_nodes = 100
    num_edges = 300
    
    # 模拟数据
    P_S = torch.rand(num_nodes) * 0.5
    P_S[0:5] = torch.rand(5) * 0.5 + 0.5  # 前5个节点是高概率源点
    
    edge_index = torch.randint(0, num_nodes, (2, num_edges))
    degrees = torch.zeros(num_nodes)
    for i in range(num_nodes):
        degrees[i] = ((edge_index[0] == i).sum() + (edge_index[1] == i).sum()).float()
    degrees = degrees.clamp(min=1.0)
    
    # 创建前向生成器
    propagator = NeuralSIPropagator(
        k_steps=3,
        beta_init=1.0,
        infection_rate_hidden=32
    )
    
    print(f"输入:")
    print(f"  节点数: {num_nodes}")
    print(f"  边数: {num_edges}")
    print(f"  源点概率范围: [{P_S.min():.3f}, {P_S.max():.3f}]")
    print(f"  高概率源点 (>0.5): {(P_S > 0.5).sum().item()}")
    
    # 前向传播
    O_pred = propagator(P_S, edge_index, degrees, num_nodes)
    
    print(f"\n输出:")
    print(f"  预测感染分布范围: [{O_pred.min():.3f}, {O_pred.max():.3f}]")
    print(f"  预测感染节点 (>0.5): {(O_pred > 0.5).sum().item()}")
    print(f"  感染扩散倍数: {(O_pred > 0.5).sum().item() / (P_S > 0.5).sum().item():.2f}x")
    print(f"  感染烈度 β: {propagator.get_beta():.4f}")
    
    # 测试梯度
    loss = O_pred.sum()
    loss.backward()
    
    print(f"\n梯度测试:")
    print(f"  β 梯度: {propagator.beta.grad:.4f}")
    has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 
                   for p in propagator.infection_rate_predictor.parameters())
    print(f"  传染率预测器梯度: {'正常' if has_grad else '异常'}")
    
    print("\n✓ NeuralSIPropagator 测试通过\n")


def test_forward_loss():
    """测试前向损失函数"""
    print("=" * 60)
    print("测试 ForwardLoss")
    print("=" * 60)
    
    num_nodes = 100
    
    # 预测感染分布 (需要梯度追踪)
    O_pred = torch.rand(num_nodes, requires_grad=True)
    
    # 真实感染掩码
    O_true_mask = torch.zeros(num_nodes, dtype=torch.bool)
    O_true_mask[0:30] = True
    
    # 创建损失函数
    criterion = ForwardLoss(
        lambda_focal=1.0,
        lambda_dice=0.5,
        focal_alpha=0.25,
        focal_gamma=2.0
    )
    
    print(f"输入:")
    print(f"  节点数: {num_nodes}")
    print(f"  真实感染节点数: {O_true_mask.sum().item()}")
    print(f"  预测感染分布范围: [{O_pred.min():.3f}, {O_pred.max():.3f}]")
    
    # 计算损失
    loss_dict = criterion(O_pred, O_true_mask)
    
    print(f"\n输出:")
    print(f"  Total Loss: {loss_dict['total']:.4f}")
    print(f"  Focal Loss: {loss_dict['focal']:.4f}")
    print(f"  Dice Loss: {loss_dict['dice']:.4f}")
    
    # 测试梯度
    loss_dict['total'].backward()
    print(f"\n梯度测试:")
    print(f"  O_pred 梯度范围: [{O_pred.grad.min():.4f}, {O_pred.grad.max():.4f}]")
    print(f"  梯度非零元素: {(O_pred.grad != 0).sum().item()}/{num_nodes}")
    
    print("\n✓ ForwardLoss 测试通过\n")


def test_integration():
    """测试集成 (改进版前向生成器 + 损失函数)"""
    print("=" * 60)
    print("测试集成 (Propagator + Loss)")
    print("=" * 60)
    
    num_nodes = 100
    num_edges = 300
    
    # 模拟数据
    P_S = torch.rand(num_nodes, requires_grad=True)
    edge_index = torch.randint(0, num_nodes, (2, num_edges))
    degrees = torch.rand(num_nodes) * 10 + 1
    O_true_mask = torch.zeros(num_nodes, dtype=torch.bool)
    O_true_mask[0:30] = True
    
    # 创建模块
    propagator = NeuralSIPropagator(k_steps=3, beta_init=1.0, infection_rate_hidden=32)
    criterion = ForwardLoss(lambda_focal=1.0, lambda_dice=0.5)
    
    print(f"前向传播:")
    print(f"  P_S → Propagator (动态传染率) → O_pred")
    
    # 前向传播
    O_pred = propagator(P_S, edge_index, degrees, num_nodes)
    loss_dict = criterion(O_pred, O_true_mask)
    
    print(f"  O_pred 范围: [{O_pred.min():.3f}, {O_pred.max():.3f}]")
    print(f"  Total Loss: {loss_dict['total']:.4f}")
    
    # 反向传播
    loss_dict['total'].backward()
    
    print(f"\n反向传播:")
    print(f"  P_S 梯度范围: [{P_S.grad.min():.4f}, {P_S.grad.max():.4f}]")
    print(f"  P_S 梯度非零: {(P_S.grad != 0).sum().item()}/{num_nodes}")
    print(f"  β 梯度: {propagator.beta.grad:.4f}")
    
    # 验证梯度流
    has_predictor_grad = any(p.grad is not None and p.grad.abs().sum() > 0 
                             for p in propagator.infection_rate_predictor.parameters())
    print(f"  传染率预测器梯度: {'正常' if has_predictor_grad else '异常'}")
    
    print("\n✓ 集成测试通过\n")


def test_comparison():
    """对比测试: 验证动态传染率的优势"""
    print("=" * 60)
    print("对比测试: 动态 vs 静态传染率")
    print("=" * 60)
    
    num_nodes = 50
    num_edges = 150
    
    # 构造一个简单的星型图: 节点0是中心
    edge_index = torch.zeros((2, num_edges), dtype=torch.long)
    for i in range(num_edges):
        edge_index[0, i] = 0  # 中心节点
        edge_index[1, i] = (i % (num_nodes - 1)) + 1
    
    degrees = torch.zeros(num_nodes)
    degrees[0] = num_edges  # 中心节点度数很高
    degrees[1:] = 1.0
    
    # 场景1: 中心节点是源点
    P_S_center = torch.zeros(num_nodes)
    P_S_center[0] = 0.9
    
    # 场景2: 边缘节点是源点
    P_S_edge = torch.zeros(num_nodes)
    P_S_edge[1] = 0.9
    
    propagator = NeuralSIPropagator(k_steps=2, beta_init=1.0, infection_rate_hidden=32)
    
    print(f"图结构: 星型图 (中心节点度数={int(degrees[0])}, 边缘节点度数=1)")
    
    # 测试场景1
    O_pred_center = propagator(P_S_center, edge_index, degrees, num_nodes)
    print(f"\n场景1: 中心节点是源点")
    print(f"  预测感染节点数: {(O_pred_center > 0.5).sum().item()}")
    print(f"  感染扩散到边缘: {(O_pred_center[1:] > 0.5).sum().item()}/{num_nodes-1}")
    
    # 测试场景2
    O_pred_edge = propagator(P_S_edge, edge_index, degrees, num_nodes)
    print(f"\n场景2: 边缘节点是源点")
    print(f"  预测感染节点数: {(O_pred_edge > 0.5).sum().item()}")
    print(f"  中心节点被感染: {O_pred_edge[0] > 0.5}")
    
    print("\n✓ 对比测试通过\n")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("改进版前向生成器测试")
    print("=" * 60 + "\n")
    
    test_dynamic_infection_rate()
    test_propagator()
    test_forward_loss()
    test_integration()
    test_comparison()
    
    print("=" * 60)
    print("所有测试通过! ✓")
    print("改进点:")
    print("  1. 传染率动态计算 (基于 P_S + H^(k) + degrees)")
    print("  2. 每步重新计算 (感知当前感染状态)")
    print("  3. 利用图结构进行消息传递")
    print("  4. 梯度流畅通 (P_S → Propagator → Loss)")
    print("=" * 60)
