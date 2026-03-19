"""
GFRR 模型包

包含:
    - encoder_gfrr: GFRR 编码器 (双通道输入 + GAT)
    - gfrr: GFRR 主模型
"""
from .encoder_gfrr import GFRREncoder, DualChannelInput, GATv2Layer
from .gfrr import GFRRLite, ClassificationHead

__all__ = [
    'GFRREncoder',
    'DualChannelInput',
    'GATv2Layer',
    'GFRRLite',
    'ClassificationHead'
]
