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
NUM_FEATURES = 22  # 特征维度（14个基础特征 + 8个最大CC内部相对特征）
TRAIN_RATIO = 0.6
VAL_RATIO = 0.2
TEST_RATIO = 0.2

# ===== 评估配置 =====
RECALL_K_VALUES = [5, 15, 25]  # 计算 Recall@5, Recall@15, Recall@25

# ===== 最大CC输入图配置 =====
# 说明:
#   - USE_MAX_CC_GRAPH=True 时, 每个样本直接使用感染子图的最大CC诱导子图作为 GNN 输入
#   - 当前默认策略为“全图输入 + 最大CC内主监督”，因此不直接裁图
#   - “源点100%落在最大CC”的结论来自最终快照, 默认只保留最终快照
USE_MAX_CC_GRAPH = False
MAX_CC_GRAPH_FINAL_ONLY = True
SKIP_SAMPLES_WITH_MISSING_SOURCES = True

# ===== 最大CC硬门控配置 =====
# 当前主策略依赖训练目标聚焦最大CC内部, 默认不做推理期硬裁剪
HARD_GATE_MAX_CC = False
LOGIT_GATE_VALUE = 12.0   # 对最大CC外感染节点的logit下压值（越大越硬）

# ===== 最大CC Pool + 回注配置 =====
# 当前主实验聚焦“最大CC内部差异化定位”，默认关闭共享池化回注，避免削弱节点间差异
USE_MAX_CC_POOL = False
MAX_CC_POOL_USE_MLP = True        # 是否对 z_cc_max 使用小MLP
MAX_CC_POOL_ALPHA = 0.20          # 最大CC内回注强度
MAX_CC_POOL_OUTSIDE_ALPHA = 0.00  # 最大CC外回注强度（建议 0 或很小）

# ===== 最大CC聚焦训练配置 =====
MAX_CC_OUTSIDE_BCE_WEIGHT = 0.20    # 最大CC外感染节点作为弱监督的BCE权重
MAX_CC_MAX_RANK_POSITIVES = 4       # 每个样本用于精排的难正样本数上限
MAX_CC_HARD_NEGATIVE_TOPK = 8       # 每个样本选取的最大CC内部 hardest negatives 数量

# ===== 排序导向模型选择 =====
MODEL_SELECTION_RECALL_K = 5
MODEL_SELECTION_WEIGHTS = {
    'map': 0.35,
    'p@k_true': 0.30,
    'recall@k': 0.20,
    'f1': 0.15,
    'aed_gain': 0.10
}


# ===== GFRR (GAT-Flow Residual Refiner) 模型架构配置 =====
# 各数据集的模型架构参数（独立于训练超参数）
GFRR_ARCH_CONFIGS = {
    'android': {
        'hidden_dim': 64,          # 隐藏层维度
        'encoder_blocks': 2,       # Encoder GAT 块数量
        'dropout': 0.3,            # Dropout 比例
    },
    'christianity': {
        'hidden_dim': 32,
        'encoder_blocks': 2,       # 较少的块 (源点更显著)
        'dropout': 0.25,
    },
    'douban': {
        'hidden_dim': 64,
        'encoder_blocks': 2,
        'dropout': 0.3,
    },
    'twitter': {
        'hidden_dim': 64,
        'encoder_blocks': 3,
        'dropout': 0.3,
    }
}

# ===== GFRR 损失函数配置 =====
# 各数据集的损失函数权重（基于数据集特性调优）
GFRR_LOSS_CONFIGS = {
    'android': {
        'lambda_rank': 2.5,        # Ranking Loss 权重
        'margin': 0.5,             # Ranking Loss margin
        'pos_weight': 2.0,         # 静态正样本权重 (保守策略)
    },
    'christianity': {
        'lambda_rank': 0.5,
        'margin': 0.2,
        'pos_weight': 5.43,        # 接近真实正负样本比 (~1:5)
    },
    'douban': {
        'lambda_rank': 1.8,
        'margin': 0.3,
        'pos_weight': 2.5,
    },
    'twitter': {
        'lambda_rank': 1.0,
        'margin': 0.3,
        'pos_weight': 2.5,         # 源点较密集 (~1:4)
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

