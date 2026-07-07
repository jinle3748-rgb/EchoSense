"""
人声分析主模块

功能：
- 整合声学特征提取、并行处理、可视化
- 提供完整的音频分析流程
- 支持单文件和批量处理
- 生成综合分析报告
- 完整的日志记录系统
"""

import os
import time
import numpy as np
import librosa
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')

from .acoustic_features import AcousticFeatureExtractor
from .parallel_processor import ParallelProcessor
from .logger import (
    setup_logger,
    log_analysis_start,
    log_analysis_complete,
    log_error,
    log_performance,
    log_features,
    log_emotion_distribution,
    log_parallel_info
)


class VoiceAnalyzer:
    """人声分析器"""
    
    def __init__(self, use_parallel=True, max_workers=None, output_dir=None, log_dir=None):
        """
        初始化人声分析器
        
        Args:
            use_parallel: 是否使用并行处理
            max_workers: 最大工作进程数
            output_dir: 输出目录
            log_dir: 日志存储目录
        """
        self.logger = setup_logger('VoiceAnalyzer', log_dir=log_dir)
        
        self.feature_extractor = AcousticFeatureExtractor()
        self.parallel_processor = ParallelProcessor(max_workers=max_workers)
        
        self.use_parallel = use_parallel
        self.output_dir = output_dir if output_dir else os.getcwd()
        
        self.window_seconds = 1.0
        self.step_seconds = 0.5
        self.top_db_limit = 35
        
        log_parallel_info(self.logger, use_parallel, self.parallel_processor.max_workers)
        
        print("\n" + "=" * 60)
        print("人声分析器初始化完成")
        print("=" * 60)
        print(f"  并行处理: {'启用' if use_parallel else '禁用'}")
        print(f"  输出目录: {self.output_dir}")
        print("=" * 60 + "\n")
    
    def analyze_audio(self, audio_file, generate_report=True):
        """
        分析单个音频文件
        
        Args:
            audio_file: 音频文件路径
            generate_report: 是否生成报告
        
        Returns:
            dict: 分析结果
        """
        log_analysis_start(self.logger, audio_file)
        print("\n" + "=" * 60)
        print(f"开始分析音频: {os.path.basename(audio_file)}")
        print("=" * 60)
        
        if not os.path.exists(audio_file):
            error_msg = f"文件不存在 - {audio_file}"
            print(f"错误: {error_msg}")
            log_error(self.logger, FileNotFoundError(error_msg), "加载音频文件")
            return None
        
        start_time = time.time()
        
        try:
            self.logger.info("[步骤1] 加载音频文件")
            print("\n[步骤1] 加载音频文件...")
            wav, sr = librosa.load(audio_file, sr=None)
            print(f"  音频长度: {len(wav)/sr:.2f}秒")
            print(f"  采样率: {sr}Hz")
            self.logger.info(f"  音频长度: {len(wav)/sr:.2f}秒, 采样率: {sr}Hz")
            
            self.logger.info("[步骤2] 分割音频段")
            print("\n[步骤2] 分割音频段...")
            segments, segment_info = self._segment_audio(wav, sr)
            print(f"  总段数: {len(segments)}")
            self.logger.info(f"  分割完成，总段数: {len(segments)}")
            
            self.logger.info("[步骤3] 提取声学特征")
            print("\n[步骤3] 提取声学特征...")
            features_start = time.time()
            if self.use_parallel:
                features_timeline, features_time = self._extract_features_parallel(segments, sr)
            else:
                features_timeline, features_time = self._extract_features_serial(segments, sr)
            print(f"  特征提取耗时: {features_time:.2f}秒")
            log_performance(self.logger, "特征提取", features_time)
            
            self.logger.info("[步骤4] 聚合特征")
            print("\n[步骤4] 聚合特征...")
            overall_features = self._aggregate_features(features_timeline)
            log_features(self.logger, overall_features)
            
            total_time = time.time() - start_time
            print(f"\n分析完成，总耗时: {total_time:.2f}秒")
            log_performance(self.logger, "完整分析", total_time)
            
            result = {
                'audio_file': audio_file,
                'duration': len(wav)/sr,
                'sample_rate': sr,
                'num_segments': len(segments),
                'overall_features': overall_features,
                'features_timeline': features_timeline,
                'analysis_time': total_time,
                'features_extraction_time': features_time
            }
            
            log_analysis_complete(self.logger, result)
            
            return result
            
        except Exception as e:
            log_error(self.logger, e, "分析音频文件")
            print(f"\n错误: {str(e)}")
            return None
    
    def _segment_audio(self, wav, sr):
        """
        分割音频
        
        Args:
            wav: 音频信号
            sr: 采样率
            
        Returns:
            tuple: (音频段列表, 段信息列表)
        """
        intervals = librosa.effects.split(wav, top_db=self.top_db_limit)
        
        window_size = int(self.window_seconds * sr)
        step_size = int(self.step_seconds * sr)
        
        segments = []
        segment_info = []
        
        for start_i, end_i in intervals:
            section = wav[start_i:end_i]
            
            if len(section) >= window_size:
                for start in range(0, len(section) - window_size, step_size):
                    segment = section[start:start + window_size]
                    segments.append(segment)
                    segment_info.append({
                        'start_time': (start_i + start) / sr,
                        'end_time': (start_i + start + window_size) / sr,
                        'duration': self.window_seconds
                    })
        
        return segments, segment_info
    
    def _extract_features_parallel(self, segments, sr):
        """
        并行提取特征
        
        Args:
            segments: 音频段列表
            sr: 采样率
            
        Returns:
            tuple: (特征列表, 耗时)
        """
        process_func = self.feature_extractor.extract_segment_features
        
        results, elapsed_time = self.parallel_processor.process_segments_parallel(
            process_func, segments, sr
        )
        
        return results, elapsed_time
    
    def _extract_features_serial(self, segments, sr):
        """
        串行提取特征
        
        Args:
            segments: 音频段列表
            sr: 采样率
            
        Returns:
            tuple: (特征列表, 耗时)
        """
        process_func = self.feature_extractor.extract_segment_features
        
        results, elapsed_time = self.parallel_processor.process_segments_serial(
            process_func, segments, sr
        )
        
        return results, elapsed_time
    
    def _aggregate_features(self, features_timeline):
        """
        聚合特征
        
        Args:
            features_timeline: 特征时间线
            
        Returns:
            dict: 聚合后的特征
        """
        aggregated = {}
        
        feature_categories = ['f0', 'formants', 'spectral', 'energy', 'mfcc', 'quality', 'temporal']
        
        for category in feature_categories:
            category_features = {}
            
            category_data = [f.get(category, {}) for f in features_timeline]
            
            if len(category_data) > 0:
                all_keys = set()
                for d in category_data:
                    all_keys.update(d.keys())
                
                for key in all_keys:
                    values = [d.get(key, 0) for d in category_data]
                    valid_values = [v for v in values if v != 0]
                    
                    if len(valid_values) > 0:
                        category_features[f'{key}_mean'] = float(np.mean(valid_values))
                        category_features[f'{key}_std'] = float(np.std(valid_values))
                        category_features[f'{key}_min'] = float(np.min(valid_values))
                        category_features[f'{key}_max'] = float(np.max(valid_values))
            
            aggregated[category] = category_features
        
        return aggregated
    
    def analyze_batch(self, audio_files, generate_reports=True):
        """
        批量分析音频文件
        
        Args:
            audio_files: 音频文件列表
            generate_reports: 是否生成报告
            
        Returns:
            list: 分析结果列表
        """
        print("\n" + "=" * 60)
        print(f"开始批量分析 {len(audio_files)} 个音频文件")
        print("=" * 60)
        
        results = []
        
        for audio_file in tqdm(audio_files, desc="批量分析进度", unit="文件"):
            result = self.analyze_audio(audio_file, generate_report=generate_reports)
            results.append(result)
        
        print("\n" + "=" * 60)
        print("批量分析完成")
        print("=" * 60)
        
        return results
    
    def compare_parallel_performance(self, audio_file):
        """
        对比并行和串行性能
        
        Args:
            audio_file: 音频文件
            
        Returns:
            dict: 性能对比结果
        """
        print("\n" + "=" * 60)
        print("并行 vs 串行 性能对比测试")
        print("=" * 60)
        
        wav, sr = librosa.load(audio_file, sr=None)
        segments, _ = self._segment_audio(wav, sr)
        
        process_func = self.feature_extractor.extract_segment_features
        
        performance_result = self.parallel_processor.compare_parallel_serial(
            process_func, segments, sr
        )
        
        return performance_result
    
    def get_feature_summary(self, result):
        """
        获取特征摘要
        
        Args:
            result: 分析结果
            
        Returns:
            str: 特征摘要文本
        """
        features = result.get('overall_features', {})
        emotion = result.get('emotion_result', {})
        
        summary = f"""
人声分析摘要
{'=' * 40}

音频信息:
  文件: {os.path.basename(result.get('audio_file', 'Unknown'))}
  时长: {result.get('duration', 0):.2f}秒
  采样率: {result.get('sample_rate', 0)}Hz

声学特征:
  平均基频: {features.get('f0', {}).get('mean_f0_mean', 0):.1f}Hz
  基频范围: {features.get('f0', {}).get('range_f0_mean', 0):.1f}Hz
  频谱质心: {features.get('spectral', {}).get('spectral_centroid_mean_mean', 0):.1f}Hz
  RMS能量: {features.get('energy', {}).get('rms_mean_mean', 0):.4f}

语音质量:
  HNR: {features.get('quality', {}).get('hnr_mean', 0):.1f}dB
  Jitter: {features.get('quality', {}).get('jitter_mean', 0):.2f}%
  Shimmer: {features.get('quality', {}).get('shimmer_mean', 0):.2f}%

情绪分析:
  主导情绪: {emotion.get('dominant_emotion', 'Unknown')}
  置信度: {emotion.get('confidence', 0):.2%}
  强度: {emotion.get('intensity', 0):.2%}

分析耗时: {result.get('analysis_time', 0):.2f}秒
{'=' * 40}
"""
        
        return summary
    
    def export_results(self, result, output_file=None):
        """
        导出分析结果
        
        Args:
            result: 分析结果
            output_file: 输出文件路径
        
        Returns:
            str: 输出文件路径
        """
        import json
        
        if output_file is None:
            audio_file = result.get('audio_file', 'unknown')
            output_file = os.path.join(
                self.output_dir,
                f'{os.path.splitext(os.path.basename(audio_file))[0]}_analysis_result.json'
            )
        
        self.logger.info(f"开始导出结果到: {output_file}")
        
        export_data = {
            'audio_file': result.get('audio_file'),
            'duration': result.get('duration'),
            'sample_rate': result.get('sample_rate'),
            'num_segments': result.get('num_segments'),
            'overall_features': result.get('overall_features'),
            'emotion_result': result.get('emotion_result'),
            'timeline_summary': {
                'dominant_overall': result.get('timeline_result', {}).get('dominant_overall'),
                'distribution': result.get('timeline_result', {}).get('distribution'),
                'num_changes': result.get('timeline_result', {}).get('num_changes'),
                'avg_intensity': result.get('timeline_result', {}).get('avg_intensity'),
                'avg_confidence': result.get('timeline_result', {}).get('avg_confidence')
            },
            'analysis_time': result.get('analysis_time'),
            'report_paths': result.get('report_paths')
        }
        
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, ensure_ascii=False, indent=2)
            
            self.logger.info(f"结果导出成功: {output_file}")
            print(f"分析结果已导出: {output_file}")
            return output_file
        except Exception as e:
            log_error(self.logger, e, "导出分析结果")
            raise
    
    def analyze_batch(self, audio_files, generate_reports=True):
        """
        批量分析音频文件
        
        Args:
            audio_files: 音频文件列表
            generate_reports: 是否生成报告
        
        Returns:
            list: 分析结果列表
        """
        self.logger.info(f"开始批量分析，共 {len(audio_files)} 个文件")
        print("\n" + "=" * 60)
        print(f"开始批量分析 {len(audio_files)} 个音频文件")
        print("=" * 60)
        
        results = []
        successful = 0
        failed = 0
        
        for idx, audio_file in enumerate(audio_files, 1):
            try:
                self.logger.info(f"[{idx}/{len(audio_files)}] 处理: {os.path.basename(audio_file)}")
                result = self.analyze_audio(audio_file, generate_report=generate_reports)
                if result:
                    results.append(result)
                    successful += 1
                else:
                    failed += 1
            except Exception as e:
                log_error(self.logger, e, f"批量处理文件: {audio_file}")
                failed += 1
        
        self.logger.info(f"批量分析完成: 成功={successful}, 失败={failed}")
        print("\n" + "=" * 60)
        print("批量分析完成")
        print("=" * 60)
        print(f"  成功: {successful} 个")
        print(f"  失败: {failed} 个")
        
        return results


def main():
    """主函数示例"""
    import sys
    
    analyzer = VoiceAnalyzer(use_parallel=True)
    
    test_dir = os.path.join(os.path.dirname(__file__), '..', 'testvoices')
    
    audio_files = []
    for root, dirs, files in os.walk(test_dir):
        for file in files:
            if file.endswith('.wav') or file.endswith('.mp3'):
                audio_files.append(os.path.join(root, file))
    
    if len(audio_files) == 0:
        print("未找到测试音频文件")
        return
    
    test_file = audio_files[0]
    print(f"使用测试文件: {test_file}")
    
    result = analyzer.analyze_audio(test_file, generate_report=True)
    
    if result:
        print(analyzer.get_feature_summary(result))
        
        analyzer.export_results(result)
        
        if analyzer.use_parallel:
            performance = analyzer.compare_parallel_performance(test_file)
            print(f"\n加速比: {performance['speedup']:.2f}x")


if __name__ == "__main__":
    main()