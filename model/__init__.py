"""
GFRR 模型包

包含:
    - encoder_gfrr: GFRR 编码器 (双通道输入 + GAT)
    - gfrr: GFRR 主模型
    - neural_si_propagator: 前向生成器 (可微神经 SI 传播器)
"""
from .encoder_gfrr import GFRREncoder, DualChannelInput, PIRALayer
from .gfrr import GFRRLite, ClassificationHead
from .neural_si_propagator import NeuralSIPropagator, DynamicInfectionRatePredictor

__all__ = [
    'GFRREncoder',
    'DualChannelInput',
    'PIRALayer',
    'GFRRLite',
    'ClassificationHead',
    'NeuralSIPropagator',
    'DynamicInfectionRatePredictor'
]
