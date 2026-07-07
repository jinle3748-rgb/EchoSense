"""
EmotionAnalysis - 情绪分析模块

功能：
- 基于声学特征的情绪识别
- 情绪强度分析
- 情绪变化趋势分析
- 多维度情绪评估
- 情绪可视化
- 情绪数据导出
"""

from .emotion_analyzer import EmotionAnalyzer
from .emotion_visualizer import EmotionVisualizer

__all__ = [
    'EmotionAnalyzer',
    'EmotionVisualizer'
]

__version__ = '1.0.0'
