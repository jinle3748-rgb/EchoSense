"""
Gender module - 性别识别模块

功能：
- GPU加速的性别识别
- 基于声学特征的性别判断
"""

from .GPU_GenderRecognition import GPUGenderAnalyzer
from .GenderRecognition import AdvancedGenderAnalyzer
from .ParallelGenderRecognition import ParallelGenderAnalyzer

__all__ = [
    'GPUGenderAnalyzer',
    'AdvancedGenderAnalyzer',
    'ParallelGenderAnalyzer'
]
