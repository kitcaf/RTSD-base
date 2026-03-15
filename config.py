"""
GFRR 模型配置文件
"""
import torch

# ===== 全局配置 =====
SEED = 3407
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ===== 数据集配置 =====
DATASETS = ['android_25c.SG', 'christianity_25c.SG', 'douban_25c.SG', 'twitter_25c.SG']
DATASET_NAMES = ['android', 'christianity', 'douban', 'twitter']
DATASET_IDX = 0  # 修改此处选择数据集: 0=android, 1=christianity, 2=douban, 3=twitter

# ===== 统一训练超参数 =====
EPOCHS = 100
LR = 0.005
BATCH_SIZE = 1
WEIGHT_DECAY = 5e-4

# ===== 通用特征和数据配置 =====
NUM_FEATURES = 14  # 特征维度
TRAIN_RATIO = 0.6
VAL_RATIO = 0.2
TEST_RATIO = 0.2

# ===== 评估配置 =====
RECALL_K_VALUES = [5, 15, 25]  # 计算 Recall@5, Recall@15, Recall@25


# ===== GFRR (GAT-Flow Residual Refiner) 模型架构配置 =====
# 各数据集的模型架构参数（独立于训练超参数）
GFRR_ARCH_CONFIGS = {
    'android': {
        'hidden_dim': 64,          # 隐藏层维度
        'encoder_blocks': 2,       # Encoder GAT 块数量
        'dropout': 0.3,            # Dropout 比例
        'k_hop': 2,                # K-hop 虚拟图聚合层数
        'lambda_1': 0.5,           # 物理距离衰减系数
        'lambda_2': 1.0,           # 大V防虹吸度数惩罚系数
        'beta': 1.0,               # PIRA 感染梯度引导系数
    },
    'christianity': {
        'hidden_dim': 32,
        'encoder_blocks': 2,       # 较少的块 (源点更显著)
        'dropout': 0.25,
        'k_hop': 2,
        'lambda_1': 0.5,
        'lambda_2': 1.0,
        'beta': 1.0,
    },
    'douban': {
        'hidden_dim': 64,
        'encoder_blocks': 2,
        'dropout': 0.3,
        'k_hop': 2,
        'lambda_1': 1.0,           # douban 梯度平原更平，稍微加强距离衰减使得跨跳更理智
        'lambda_2': 1.0,
        'beta': 2.0,               # 强化梯度引导跨越平原
    },
    'twitter': {
        'hidden_dim': 64,
        'encoder_blocks': 3,
        'dropout': 0.3,
        'k_hop': 2,
        'lambda_1': 0.5,
        'lambda_2': 2.0,           # twitter 度分布极不平衡，加强大V惩罚
        'beta': 1.0,
    }
}

# ===== GFRR 损失函数配置 =====
# 各数据集的损失函数权重（基于数据集特性调优）
GFRR_LOSS_CONFIGS = {
    'android': {
        'lambda_rank': 2.0,        # Ranking Loss 权重
        'margin': 0.5,             # Ranking Loss margin
        'pos_weight': 2.0,         # 静态正样本权重 (保守策略)
        'lambda_cc': 0.2,          # CC-Contrastive Loss 权重
        'temperature': 0.1,        # 对比学习温度参数
    },
    'christianity': {
        'lambda_rank': 0.2,
        'margin': 0.2,
        'pos_weight': 5.43,        # 接近真实正负样本比 (~1:5)
        'lambda_cc': 0.15,         # Christianity连通性好，CC对比权重可稍低
        'temperature': 0.1,
    },
    'douban': {
        'lambda_rank': 1.5,
        'margin': 0.3,
        'pos_weight': 2.5,
        'lambda_cc': 0.25,         # Douban碎片化严重，CC对比权重稍高
        'temperature': 0.1,
    },
    'twitter': {
        'lambda_rank': 0.8,
        'margin': 0.3,
        'pos_weight': 2.5,         # 源点较密集 (~1:4)
        'lambda_cc': 0.2,
        'temperature': 0.1,
    }
}

# ===== GFRR 动态 pos_weight 配置 =====
USE_DYNAMIC_POS_WEIGHT = False  # 是否启用动态 pos_weight (建议 False 以使用特调静态权重)
POS_WEIGHT_SCALE = 1.0         # 缩放因子 (仅在启用动态时有效)
POS_WEIGHT_MIN = 2.0           # 最小权重
POS_WEIGHT_MAX = 50.0          # 最大权重
DEFAULT_POS_WEIGHT = 2.0       # Fallback 默认权重



# ==================== 辅助函数 ====================
def get_gfrr_arch_config():
    """获取当前数据集的 GFRR 架构配置"""
    return GFRR_ARCH_CONFIGS[DATASET_NAMES[DATASET_IDX]]

def get_gfrr_loss_config():
    """获取当前数据集的 GFRR 损失函数配置"""
    return GFRR_LOSS_CONFIGS[DATASET_NAMES[DATASET_IDX]]
