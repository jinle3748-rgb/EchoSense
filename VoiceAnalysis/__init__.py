"""
VoiceAnalysis - 人声分析模块

功能：
- 基础声学特征提取（基频、共振峰、频谱特征等）
- 情绪/情感分析（集成在VoiceAnalyzer中）
- 并行计算优化
- 可视化报告生成
- 完整的日志记录系统
"""

from .acoustic_features import AcousticFeatureExtractor
from .voice_analyzer import VoiceAnalyzer
from .parallel_processor import ParallelProcessor
from .logger import (
    setup_logger,
    log_analysis_start,
    log_analysis_complete,
    log_error,
    log_performance,
    log_features,
    log_emotion_distribution,
    log_gui_operation,
    log_parallel_info
)

__all__ = [
    'AcousticFeatureExtractor',
    'VoiceAnalyzer',
    'ParallelProcessor',
    'setup_logger',
    'log_analysis_start',
    'log_analysis_complete',
    'log_error',
    'log_performance',
    'log_features',
    'log_emotion_distribution',
    'log_gui_operation',
    'log_parallel_info'
]

__version__ = '2.0.0'