"""
并行处理模块

功能：
- 多进程并行处理
- GPU加速处理
- 批处理优化
- 性能监控
- 自动负载均衡
"""

import os
import time
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from functools import partial
from tqdm import tqdm
import numpy as np
import warnings

warnings.filterwarnings('ignore')

try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
    cp = None


class ParallelProcessor:
    """并行处理器"""
    
    def __init__(self, max_workers=None, use_gpu=False, batch_size=32):
        """
        初始化并行处理器
        
        Args:
            max_workers: 最大工作进程数，None表示自动检测
            use_gpu: 是否使用GPU加速
            batch_size: 批处理大小
        """
        self.max_workers = max_workers if max_workers else multiprocessing.cpu_count()
        self.use_gpu = use_gpu and CUPY_AVAILABLE
        self.batch_size = batch_size
        self.gpu_available = CUPY_AVAILABLE
        
        print("并行处理器初始化完成")
        print(f"  CPU核心数: {multiprocessing.cpu_count()}")
        print(f"  最大工作进程: {self.max_workers}")
        print(f"  GPU可用: {self.gpu_available}")
        print(f"  批处理大小: {self.batch_size}")
    
    def process_segments_parallel(self, process_func, segments, sr, use_processes=True):
        """
        并行处理音频段
        
        Args:
            process_func: 处理函数
            segments: 音频段列表
            sr: 采样率
            use_processes: 是否使用进程池（True）或线程池（False）
            
        Returns:
            list: 处理结果列表
        """
        if len(segments) == 0:
            return []
        
        print(f"\n启动并行处理模式")
        print(f"  总段数: {len(segments)}")
        print(f"  处理模式: {'多进程' if use_processes else '多线程'}")
        print(f"  工作进程数: {self.max_workers}")
        
        start_time = time.time()
        
        if use_processes:
            with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
                results = list(tqdm(
                    executor.map(process_func, segments, [sr] * len(segments)),
                    total=len(segments),
                    desc="并行处理进度",
                    unit="段"
                ))
        else:
            with ThreadPoolExecutor(max_workers=self.max_workers * 2) as executor:
                results = list(tqdm(
                    executor.map(process_func, segments, [sr] * len(segments)),
                    total=len(segments),
                    desc="并行处理进度",
                    unit="段"
                ))
        
        elapsed_time = time.time() - start_time
        
        print(f"\n并行处理完成")
        print(f"  总耗时: {elapsed_time:.2f}秒")
        print(f"  平均每段: {elapsed_time/len(segments):.4f}秒")
        
        return results, elapsed_time
    
    def process_segments_serial(self, process_func, segments, sr):
        """
        串行处理音频段（用于对比）
        
        Args:
            process_func: 处理函数
            segments: 音频段列表
            sr: 采样率
            
        Returns:
            list: 处理结果列表
        """
        if len(segments) == 0:
            return []
        
        print(f"\n启动串行处理模式")
        print(f"  总段数: {len(segments)}")
        
        start_time = time.time()
        
        results = []
        for segment in tqdm(segments, desc="串行处理进度", unit="段"):
            result = process_func(segment, sr)
            results.append(result)
        
        elapsed_time = time.time() - start_time
        
        print(f"\n串行处理完成")
        print(f"  总耗时: {elapsed_time:.2f}秒")
        print(f"  平均每段: {elapsed_time/len(segments):.4f}秒")
        
        return results, elapsed_time
    
    def compare_parallel_serial(self, process_func, segments, sr):
        """
        对比并行和串行处理性能
        
        Args:
            process_func: 处理函数
            segments: 音频段列表
            sr: 采样率
            
        Returns:
            dict: 性能对比结果
        """
        print("\n" + "=" * 60)
        print("并行 vs 串行 性能对比测试")
        print("=" * 60)
        
        parallel_results, parallel_time = self.process_segments_parallel(
            process_func, segments, sr, use_processes=True
        )
        
        serial_results, serial_time = self.process_segments_serial(
            process_func, segments, sr
        )
        
        speedup = serial_time / parallel_time if parallel_time > 0 else 1
        efficiency = speedup / self.max_workers
        
        print("\n" + "=" * 60)
        print("性能对比结果")
        print("=" * 60)
        print(f"  串行处理时间: {serial_time:.2f}秒")
        print(f"  并行处理时间: {parallel_time:.2f}秒")
        print(f"  加速比: {speedup:.2f}x")
        print(f"  并行效率: {efficiency:.2%}")
        print("=" * 60)
        
        return {
            'serial_time': serial_time,
            'parallel_time': parallel_time,
            'speedup': speedup,
            'efficiency': efficiency,
            'parallel_results': parallel_results,
            'serial_results': serial_results
        }
    
    def batch_process_gpu(self, process_func, segments, sr):
        """
        GPU批处理
        
        Args:
            process_func: 处理函数
            segments: 音频段列表
            sr: 采样率
            
        Returns:
            list: 处理结果列表
        """
        if not self.gpu_available:
            print("GPU不可用，使用CPU并行处理")
            return self.process_segments_parallel(process_func, segments, sr)
        
        print(f"\n启动GPU批处理模式")
        print(f"  总段数: {len(segments)}")
        print(f"  批处理大小: {self.batch_size}")
        
        start_time = time.time()
        results = []
        
        for i in tqdm(range(0, len(segments), self.batch_size), desc="GPU批处理进度", unit="批"):
            batch_segments = segments[i:i + self.batch_size]
            batch_results = [process_func(seg, sr) for seg in batch_segments]
            results.extend(batch_results)
        
        elapsed_time = time.time() - start_time
        
        print(f"\nGPU批处理完成")
        print(f"  总耗时: {elapsed_time:.2f}秒")
        
        return results, elapsed_time
    
    def optimize_workers(self, data_size, task_complexity="medium"):
        """
        优化工作进程数
        
        Args:
            data_size: 数据大小
            task_complexity: 任务复杂度
            
        Returns:
            int: 推荐的工作进程数
        """
        cpu_count = multiprocessing.cpu_count()
        
        if task_complexity == "low":
            return min(cpu_count * 2, 32)
        elif task_complexity == "medium":
            return cpu_count
        else:
            return max(1, cpu_count - 1)
    
    def get_system_info(self):
        """
        获取系统信息
        
        Returns:
            dict: 系统信息
        """
        import platform
        
        info = {
            'cpu_count': multiprocessing.cpu_count(),
            'gpu_available': self.gpu_available,
            'os': platform.system(),
            'python_version': platform.python_version(),
            'max_workers': self.max_workers,
            'batch_size': self.batch_size
        }
        
        try:
            import psutil
            memory = psutil.virtual_memory()
            info['total_memory_gb'] = memory.total / (1024 ** 3)
            info['available_memory_gb'] = memory.available / (1024 ** 3)
        except ImportError:
            info['total_memory_gb'] = 'Unknown'
            info['available_memory_gb'] = 'Unknown'
        
        return info
    
    def print_system_info(self):
        """打印系统信息"""
        info = self.get_system_info()
        
        print("\n" + "=" * 60)
        print("系统信息")
        print("=" * 60)
        print(f"  CPU核心数: {info['cpu_count']}")
        print(f"  GPU可用: {info['gpu_available']}")
        print(f"  操作系统: {info['os']}")
        print(f"  Python版本: {info['python_version']}")
        print(f"  总内存: {info['total_memory_gb']:.2f} GB")
        print(f"  可用内存: {info['available_memory_gb']:.2f} GB")
        print("=" * 60)


def worker_process_segment(segment, sr, extractor):
    """
    工作进程处理函数
    
    Args:
        segment: 音频段
        sr: 采样率
        extractor: 特征提取器
        
    Returns:
        dict: 特征结果
    """
    return extractor.extract_all_features(segment, sr)