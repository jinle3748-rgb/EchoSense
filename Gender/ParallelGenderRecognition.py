#!/usr/bin/env python3
"""
并行性别识别模块

功能：
- 支持并行处理多个音频文件
- 提高批量处理效率
- 支持进度跟踪
"""

import os
import json
import numpy as np
from multiprocessing import Pool, cpu_count
from .GPU_GenderRecognition import GPUGenderAnalyzer

class ParallelGenderAnalyzer:
    """并行性别分析器"""
    
    def __init__(self, num_workers=None):
        """
        初始化并行性别分析器
        
        Args:
            num_workers: 并行工作进程数，默认为CPU核心数
        """
        self.num_workers = num_workers or max(1, cpu_count() - 1)
        self.analyzer = GPUGenderAnalyzer()
        
        print(f"Parallel Gender Analyzer initialized with {self.num_workers} workers")
    
    def _analyze_single(self, args):
        """分析单个音频文件（内部方法）"""
        audio_path, index, total = args
        result = self.analyzer.analyze(audio_path)
        result['file'] = audio_path
        result['index'] = index
        result['total'] = total
        return result
    
    def batch_analyze(self, audio_paths, callback=None):
        """
        批量并行分析多个音频文件
        
        Args:
            audio_paths: 音频文件路径列表
            callback: 进度回调函数 (optional)
            
        Returns:
            list: 分析结果列表
        """
        total_files = len(audio_paths)
        if total_files == 0:
            return []
        
        # 准备参数
        args_list = [(audio_paths[i], i + 1, total_files) for i in range(total_files)]
        
        # 创建进程池并并行处理
        with Pool(processes=self.num_workers) as pool:
            results = []
            for i, result in enumerate(pool.imap_unordered(self._analyze_single, args_list)):
                results.append(result)
                
                # 调用回调函数更新进度
                if callback:
                    progress = (i + 1) / total_files * 100
                    callback(progress, result)
        
        # 按原始顺序排序
        results.sort(key=lambda x: x['index'])
        
        return results
    
    def analyze_directory(self, directory, callback=None):
        """
        分析目录中的所有音频文件
        
        Args:
            directory: 目录路径
            callback: 进度回调函数 (optional)
            
        Returns:
            list: 分析结果列表
        """
        audio_extensions = ('.wav', '.mp3', '.flac', '.ogg')
        audio_paths = []
        
        for root, dirs, files in os.walk(directory):
            for file in files:
                if file.lower().endswith(audio_extensions):
                    audio_paths.append(os.path.join(root, file))
        
        audio_paths.sort()
        return self.batch_analyze(audio_paths, callback)
    
    def save_results(self, results, output_path):
        """
        保存批量分析结果
        
        Args:
            results: 分析结果列表
            output_path: 输出文件路径
        """
        # 统计结果
        stats = {
            'total_files': len(results),
            'male_count': sum(1 for r in results if r['gender'] == 'male'),
            'female_count': sum(1 for r in results if r['gender'] == 'female'),
            'neutral_count': sum(1 for r in results if r['gender'] == 'neutral'),
            'unknown_count': sum(1 for r in results if r['gender'] == 'unknown'),
            'error_count': sum(1 for r in results if r['gender'] == 'error')
        }
        
        output_data = {
            'stats': stats,
            'results': results
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        
        return stats
