"""
情绪可视化模块

功能：
- 情绪分布可视化
- 情绪时间线可视化
- 情绪强度图表
- 综合情绪报告
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.gridspec import GridSpec
import warnings

warnings.filterwarnings('ignore')

matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False


class EmotionVisualizer:
    """情绪可视化器"""
    
    def __init__(self, output_dir=None):
        """
        初始化可视化器
        
        Args:
            output_dir: 输出目录，None表示当前目录
        """
        self.output_dir = output_dir if output_dir else os.getcwd()
        
        self.emotion_colors = {
            'neutral': '#808080',
            'happy': '#FFD700',
            'sad': '#4169E1',
            'angry': '#FF4500',
            'fear': '#9400D3',
            'surprise': '#00CED1',
            'disgust': '#8B4513'
        }
        
        self.emotion_names_cn = {
            'neutral': '中性',
            'happy': '快乐',
            'sad': '悲伤',
            'angry': '愤怒',
            'fear': '恐惧',
            'surprise': '惊讶',
            'disgust': '厌恶'
        }
        
        print("情绪可视化器初始化完成")
        print(f"  输出目录: {self.output_dir}")
    
    def plot_emotion_distribution(self, emotion_distribution, audio_file, save_path=None):
        """
        绘制情绪分布图
        
        Args:
            emotion_distribution: 情绪分布字典
            audio_file: 音频文件名
            save_path: 保存路径
            
        Returns:
            str: 保存的文件路径
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # 饼图
        emotions = list(emotion_distribution.keys())
        values = list(emotion_distribution.values())
        colors = [self.emotion_colors[e] for e in emotions]
        emotion_labels = [self.emotion_names_cn.get(e, e) for e in emotions]
        
        ax1.pie(values, labels=emotion_labels, colors=colors, autopct='%1.1f%%', 
                startangle=90, shadow=True)
        ax1.set_title('情绪分布饼图', fontsize=14, fontweight='bold')
        
        # 柱状图
        bars = ax2.bar(emotion_labels, values, color=colors, alpha=0.8)
        ax2.set_title('情绪分布柱状图', fontsize=14, fontweight='bold')
        ax2.set_ylabel('比例')
        ax2.set_ylim([0, max(values) * 1.2])
        
        # 添加数值标签
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2%}',
                    ha='center', va='bottom')
        
        plt.tight_layout()
        
        if save_path is None:
            filename = os.path.basename(audio_file)
            save_path = os.path.join(self.output_dir, f"{os.path.splitext(filename)[0]}_emotion_distribution.png")
        
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"情绪分布图已保存至: {save_path}")
        return save_path
    
    def plot_emotion_timeline(self, emotion_timeline, audio_file, save_path=None):
        """
        绘制情绪时间线
        
        Args:
            emotion_timeline: 情绪时间线结果
            audio_file: 音频文件名
            save_path: 保存路径
            
        Returns:
            str: 保存的文件路径
        """
        timeline = emotion_timeline.get('timeline', [])
        
        if not timeline:
            print("没有情绪时间线数据")
            return None
        
        fig, axes = plt.subplots(3, 1, figsize=(16, 12), sharex=True)
        
        time_points = np.arange(len(timeline))
        
        # 主要情绪
        dominant_emotions = [t['dominant_emotion'] for t in timeline]
        emotion_values = [self._emotion_to_idx(e) for e in dominant_emotions]
        colors = [self.emotion_colors[e] for e in dominant_emotions]
        
        axes[0].scatter(time_points, emotion_values, c=colors, s=100, alpha=0.7)
        axes[0].plot(time_points, emotion_values, color='gray', alpha=0.5, linewidth=1)
        axes[0].set_yticks(range(len(self.emotion_categories)))
        axes[0].set_yticklabels([self.emotion_names_cn.get(e, e) for e in self.emotion_categories])
        axes[0].set_title('主要情绪时间线', fontsize=14, fontweight='bold')
        axes[0].grid(True, alpha=0.3)
        
        # 置信度
        confidences = [t['confidence'] for t in timeline]
        intensities = [t['intensity'] for t in timeline]
        axes[1].plot(time_points, confidences, 'b-', label='置信度', linewidth=2, alpha=0.7)
        axes[1].plot(time_points, intensities, 'r--', label='强度', linewidth=2, alpha=0.7)
        axes[1].set_title('情绪置信度与强度', fontsize=14, fontweight='bold')
        axes[1].set_ylabel('数值')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        axes[1].set_ylim([0, 1.1])
        
        # 情绪变化点
        if 'changes' in emotion_timeline and emotion_timeline['changes']:
            changes = emotion_timeline['changes']
            change_positions = [c['position'] for c in changes]
            for pos in change_positions:
                axes[2].axvline(x=pos, color='red', linestyle='--', alpha=0.5, linewidth=1)
        
        # 各情绪得分时间线
        for emotion in self.emotion_categories:
            scores = [t['emotion_scores'].get(emotion, 0) for t in timeline]
            axes[2].plot(time_points, scores, label=self.emotion_names_cn.get(emotion, emotion), 
                        color=self.emotion_colors[emotion], alpha=0.6, linewidth=1.5)
        
        axes[2].set_title('各情绪得分变化', fontsize=14, fontweight='bold')
        axes[2].set_xlabel('时间片段')
        axes[2].set_ylabel('得分')
        axes[2].legend(loc='upper right', bbox_to_anchor=(1, 1), fontsize=10)
        axes[2].grid(True, alpha=0.3)
        axes[2].set_ylim([0, 1.1])
        
        fig.suptitle(f'情绪分析时间线 - {os.path.basename(audio_file)}', 
                    fontsize=16, fontweight='bold', y=0.995)
        plt.tight_layout()
        
        if save_path is None:
            filename = os.path.basename(audio_file)
            save_path = os.path.join(self.output_dir, f"{os.path.splitext(filename)[0]}_emotion_timeline.png")
        
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"情绪时间线图已保存至: {save_path}")
        return save_path
    
    def plot_emotion_summary(self, emotion_result, audio_file, save_path=None):
        """
        绘制情绪综合报告
        
        Args:
            emotion_result: 情绪分析结果
            audio_file: 音频文件名
            save_path: 保存路径
            
        Returns:
            str: 保存的文件路径
        """
        fig = plt.figure(figsize=(16, 10))
        gs = GridSpec(2, 3, figure=fig)
        
        fig.suptitle(f'情绪分析综合报告 - {os.path.basename(audio_file)}', 
                    fontsize=18, fontweight='bold', y=0.98)
        
        # 主要情绪展示
        ax1 = fig.add_subplot(gs[0, 0])
        dominant = emotion_result.get('dominant_emotion', 'neutral')
        dominant_cn = self.emotion_names_cn.get(dominant, dominant)
        confidence = emotion_result.get('confidence', 0)
        intensity = emotion_result.get('intensity', 0)
        
        ax1.bar([dominant_cn], [confidence], color=self.emotion_colors[dominant], width=0.6)
        ax1.set_title('主要情绪', fontsize=14, fontweight='bold')
        ax1.set_ylim([0, 1])
        ax1.text(0, confidence/2, f'{confidence:.1%}', ha='center', va='center', 
                fontsize=16, fontweight='bold')
        ax1.grid(True, alpha=0.3, axis='y')
        
        # 次要情绪
        ax2 = fig.add_subplot(gs[0, 1])
        secondary = emotion_result.get('secondary_emotion', 'neutral')
        secondary_cn = self.emotion_names_cn.get(secondary, secondary)
        secondary_conf = emotion_result.get('secondary_confidence', 0)
        
        ax2.bar([secondary_cn], [secondary_conf], color=self.emotion_colors[secondary], width=0.6)
        ax2.set_title('次要情绪', fontsize=14, fontweight='bold')
        ax2.set_ylim([0, 1])
        ax2.text(0, secondary_conf/2, f'{secondary_conf:.1%}', ha='center', va='center', 
                fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='y')
        
        # 情绪强度
        ax3 = fig.add_subplot(gs[0, 2])
        intensity_colors = ['#4CAF50', '#FF9800', '#F44336']
        intensity_labels = ['轻微', '中等', '强烈']
        intensity_idx = 0 if intensity < 0.4 else 1 if intensity < 0.7 else 2
        
        ax3.bar([intensity_labels[intensity_idx]], [intensity], 
               color=intensity_colors[intensity_idx], width=0.6)
        ax3.set_title('情绪强度', fontsize=14, fontweight='bold')
        ax3.set_ylim([0, 1])
        ax3.text(0, intensity/2, f'{intensity:.2f}', ha='center', va='center', 
                fontsize=16, fontweight='bold')
        ax3.grid(True, alpha=0.3, axis='y')
        
        # 所有情绪得分
        ax4 = fig.add_subplot(gs[1, :])
        emotion_scores = emotion_result.get('emotion_scores', {})
        emotions = list(emotion_scores.keys())
        scores = list(emotion_scores.values())
        colors = [self.emotion_colors[e] for e in emotions]
        emotion_labels = [self.emotion_names_cn.get(e, e) for e in emotions]
        
        bars = ax4.bar(emotion_labels, scores, color=colors, alpha=0.8)
        ax4.set_title('各情绪得分', fontsize=14, fontweight='bold')
        ax4.set_ylabel('得分')
        ax4.set_ylim([0, max(scores) * 1.2 if scores else 1])
        
        # 添加数值标签
        for bar in bars:
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}',
                    ha='center', va='bottom', fontsize=10)
        
        ax4.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        if save_path is None:
            filename = os.path.basename(audio_file)
            save_path = os.path.join(self.output_dir, f"{os.path.splitext(filename)[0]}_emotion_summary.png")
        
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"情绪综合报告图已保存至: {save_path}")
        return save_path
    
    def _emotion_to_idx(self, emotion):
        """情绪到索引转换"""
        if emotion in self.emotion_categories:
            return self.emotion_categories.index(emotion)
        return 0
    
    @property
    def emotion_categories(self):
        """获取情绪类别列表"""
        return ['neutral', 'happy', 'sad', 'angry', 'fear', 'surprise', 'disgust']
    
    def plot_acoustic_features(self, features, audio_file, save_path=None):
        """
        绘制声学特征图
        
        Args:
            features: 声学特征字典
            audio_file: 音频文件名
            save_path: 保存路径
            
        Returns:
            str: 保存的文件路径
        """
        fig = plt.figure(figsize=(20, 12))
        gs = GridSpec(3, 3, figure=fig)
        
        fig.suptitle(f'声学特征分析 - {os.path.basename(audio_file)}', 
                    fontsize=18, fontweight='bold', y=0.98)
        
        # 基频特征
        ax1 = fig.add_subplot(gs[0, 0])
        f0 = features.get('f0', {})
        f0_keys = ['mean_f0', 'min_f0', 'max_f0', 'std_f0']
        f0_values = [f0.get(k, 0) for k in f0_keys]
        f0_labels = ['平均基频', '最低基频', '最高基频', '基频标准差']
        
        bars1 = ax1.bar(f0_labels, f0_values, color=['#4CAF50', '#8BC34A', '#CDDC39', '#FFEB3B'], alpha=0.8)
        ax1.set_title('基频特征', fontsize=12, fontweight='bold')
        ax1.tick_params(axis='x', rotation=45)
        ax1.grid(True, alpha=0.3, axis='y')
        
        # 共振峰特征
        ax2 = fig.add_subplot(gs[0, 1])
        formants = features.get('formants', {})
        formant_keys = ['f1', 'f2', 'f3']
        formant_values = [formants.get(k, 0) for k in formant_keys]
        formant_labels = ['共振峰F1', '共振峰F2', '共振峰F3']
        
        bars2 = ax2.bar(formant_labels, formant_values, color=['#2196F3', '#64B5F6', '#90CAF9'], alpha=0.8)
        ax2.set_title('共振峰特征', fontsize=12, fontweight='bold')
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(True, alpha=0.3, axis='y')
        
        # 能量特征
        ax3 = fig.add_subplot(gs[0, 2])
        energy = features.get('energy', {})
        energy_keys = ['rms_mean', 'rms_max', 'rms_min']
        energy_values = [energy.get(k, 0) for k in energy_keys]
        energy_labels = ['平均RMS能量', '最大RMS能量', '最小RMS能量']
        
        bars3 = ax3.bar(energy_labels, energy_values, color=['#F44336', '#EF5350', '#E57373'], alpha=0.8)
        ax3.set_title('能量特征', fontsize=12, fontweight='bold')
        ax3.tick_params(axis='x', rotation=45)
        ax3.grid(True, alpha=0.3, axis='y')
        
        # 语音质量特征
        ax4 = fig.add_subplot(gs[1, 0])
        quality = features.get('quality', {})
        quality_keys = ['hnr', 'jitter', 'shimmer']
        quality_values = [quality.get(k, 0) for k in quality_keys]
        quality_labels = ['谐波噪声比(HNR)', '抖动(%)', '闪烁(%)']
        
        bars4 = ax4.bar(quality_labels, quality_values, color=['#7E57C2', '#9575CD', '#B39DDB'], alpha=0.8)
        ax4.set_title('语音质量特征', fontsize=12, fontweight='bold')
        ax4.tick_params(axis='x', rotation=45)
        ax4.grid(True, alpha=0.3, axis='y')
        
        # 频谱特征
        ax5 = fig.add_subplot(gs[1, 1])
        spectral = features.get('spectral', {})
        spectral_keys = ['spectral_centroid_mean', 'spectral_bandwidth_mean', 'spectral_rolloff_85_mean']
        spectral_values = [spectral.get(k, 0) for k in spectral_keys]
        spectral_labels = ['频谱质心', '频谱带宽', '频谱滚降点']
        
        bars5 = ax5.bar(spectral_labels, spectral_values, color=['#00ACC1', '#4DD0E1', '#80DEEA'], alpha=0.8)
        ax5.set_title('频谱特征', fontsize=12, fontweight='bold')
        ax5.tick_params(axis='x', rotation=45)
        ax5.grid(True, alpha=0.3, axis='y')
        
        # 时域特征
        ax6 = fig.add_subplot(gs[1, 2])
        temporal = features.get('temporal', {})
        temporal_keys = ['voiced_ratio', 'num_voiced_segments', 'mean_segment_duration']
        temporal_values = [temporal.get(k, 0) for k in temporal_keys]
        temporal_labels = ['有声比例', '有声段数', '平均段长']
        
        bars6 = ax6.bar(temporal_labels, temporal_values, color=['#FF7043', '#FF8A65', '#FFAB91'], alpha=0.8)
        ax6.set_title('时域特征', fontsize=12, fontweight='bold')
        ax6.tick_params(axis='x', rotation=45)
        ax6.grid(True, alpha=0.3, axis='y')
        
        # MFCC特征概览
        ax7 = fig.add_subplot(gs[2, :])
        mfcc = features.get('mfcc', {})
        mfcc_means = [mfcc.get(f'mfcc_{i}_mean', 0) for i in range(13)]
        
        ax7.plot(range(1, 14), mfcc_means, 'b-', marker='o', linewidth=2, alpha=0.8)
        ax7.set_title('MFCC特征均值', fontsize=12, fontweight='bold')
        ax7.set_xlabel('MFCC系数')
        ax7.set_ylabel('均值')
        ax7.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path is None:
            filename = os.path.basename(audio_file)
            save_path = os.path.join(self.output_dir, f"{os.path.splitext(filename)[0]}_acoustic_features.png")
        
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"声学特征图已保存至: {save_path}")
        return save_path
    
    def plot_emotion_analysis(self, emotion_result, audio_file, save_path=None):
        """
        绘制情绪分析图（别名方法，兼容旧代码）
        
        Args:
            emotion_result: 情绪分析结果
            audio_file: 音频文件名
            save_path: 保存路径
            
        Returns:
            str: 保存的文件路径
        """
        return self.plot_emotion_summary(emotion_result, audio_file, save_path)
    
    def generate_comprehensive_report(self, acoustic_features, emotion_result, timeline_result, audio_file, performance_info=None, save_path=None):
        """
        生成综合分析报告
        
        Args:
            acoustic_features: 声学特征
            emotion_result: 情绪分析结果
            timeline_result: 时间线分析结果
            audio_file: 音频文件名
            performance_info: 性能信息
            save_path: 保存路径
            
        Returns:
            str: 保存的文件路径
        """
        fig = plt.figure(figsize=(24, 16))
        gs = GridSpec(4, 4, figure=fig)
        
        fig.suptitle(f'EchoSense 人声分析综合报告\n{os.path.basename(audio_file)}', 
                    fontsize=22, fontweight='bold', y=0.99)
        
        # 主要情绪展示
        ax1 = fig.add_subplot(gs[0, 0])
        dominant = emotion_result.get('dominant_emotion', 'neutral')
        dominant_cn = self.emotion_names_cn.get(dominant, dominant)
        confidence = emotion_result.get('confidence', 0)
        
        ax1.bar([dominant_cn], [confidence], color=self.emotion_colors[dominant], width=0.6)
        ax1.set_title('主要情绪', fontsize=14, fontweight='bold')
        ax1.set_ylim([0, 1.1])
        ax1.text(0, confidence/2, f'{confidence:.1%}', ha='center', va='center', 
                fontsize=20, fontweight='bold')
        ax1.grid(True, alpha=0.3, axis='y')
        
        # 情绪分布饼图
        ax2 = fig.add_subplot(gs[0, 1])
        emotion_scores = emotion_result.get('emotion_scores', {})
        emotions = list(emotion_scores.keys())
        values = list(emotion_scores.values())
        colors = [self.emotion_colors[e] for e in emotions]
        emotion_labels = [self.emotion_names_cn.get(e, e) for e in emotions]
        
        ax2.pie(values, labels=emotion_labels, colors=colors, autopct='%1.1f%%', 
                startangle=90, shadow=True)
        ax2.set_title('情绪分布', fontsize=14, fontweight='bold')
        
        # 基频信息
        ax3 = fig.add_subplot(gs[0, 2])
        f0 = acoustic_features.get('f0', {})
        f0_mean = f0.get('mean_f0', 0)
        f0_range = f0.get('range_f0', 0)
        
        ax3.bar(['平均基频(Hz)', '基频范围(Hz)'], [f0_mean, f0_range], 
               color=['#4CAF50', '#8BC34A'], alpha=0.8)
        ax3.set_title('基频信息', fontsize=14, fontweight='bold')
        ax3.grid(True, alpha=0.3, axis='y')
        
        # 共振峰信息
        ax4 = fig.add_subplot(gs[0, 3])
        formants = acoustic_features.get('formants', {})
        formant_values = [formants.get('f1', 0), formants.get('f2', 0), formants.get('f3', 0)]
        formant_labels = ['F1', 'F2', 'F3']
        
        ax4.bar(formant_labels, formant_values, color=['#2196F3', '#64B5F6', '#90CAF9'], alpha=0.8)
        ax4.set_title('共振峰(Hz)', fontsize=14, fontweight='bold')
        ax4.grid(True, alpha=0.3, axis='y')
        
        # 语音质量
        ax5 = fig.add_subplot(gs[1, 0])
        quality = acoustic_features.get('quality', {})
        quality_values = [quality.get('hnr', 0), quality.get('jitter', 0), quality.get('shimmer', 0)]
        quality_labels = ['HNR(dB)', 'Jitter(%)', 'Shimmer(%)']
        
        ax5.bar(quality_labels, quality_values, color=['#7E57C2', '#9575CD', '#B39DDB'], alpha=0.8)
        ax5.set_title('语音质量', fontsize=14, fontweight='bold')
        ax5.grid(True, alpha=0.3, axis='y')
        
        # 能量特征
        ax6 = fig.add_subplot(gs[1, 1])
        energy = acoustic_features.get('energy', {})
        energy_values = [energy.get('rms_mean', 0), energy.get('total_energy', 0)]
        energy_labels = ['RMS能量', '总能量']
        
        ax6.bar(energy_labels, energy_values, color=['#F44336', '#EF5350'], alpha=0.8)
        ax6.set_title('能量特征', fontsize=14, fontweight='bold')
        ax6.grid(True, alpha=0.3, axis='y')
        
        # 频谱特征
        ax7 = fig.add_subplot(gs[1, 2])
        spectral = acoustic_features.get('spectral', {})
        spectral_values = [spectral.get('spectral_centroid_mean', 0), 
                          spectral.get('spectral_bandwidth_mean', 0)]
        spectral_labels = ['频谱质心', '频谱带宽']
        
        ax7.bar(spectral_labels, spectral_values, color=['#00ACC1', '#4DD0E1'], alpha=0.8)
        ax7.set_title('频谱特征', fontsize=14, fontweight='bold')
        ax7.grid(True, alpha=0.3, axis='y')
        
        # 时域特征
        ax8 = fig.add_subplot(gs[1, 3])
        temporal = acoustic_features.get('temporal', {})
        temporal_values = [temporal.get('voiced_ratio', 0) * 100, 
                          temporal.get('num_voiced_segments', 0)]
        temporal_labels = ['有声比例(%)', '有声段数']
        
        ax8.bar(temporal_labels, temporal_values, color=['#FF7043', '#FF8A65'], alpha=0.8)
        ax8.set_title('时域特征', fontsize=14, fontweight='bold')
        ax8.grid(True, alpha=0.3, axis='y')
        
        # MFCC特征
        ax9 = fig.add_subplot(gs[2, :2])
        mfcc = acoustic_features.get('mfcc', {})
        mfcc_means = [mfcc.get(f'mfcc_{i}_mean', 0) for i in range(13)]
        
        ax9.plot(range(1, 14), mfcc_means, 'b-', marker='o', linewidth=2, alpha=0.8)
        ax9.set_title('MFCC特征系数', fontsize=14, fontweight='bold')
        ax9.set_xlabel('MFCC系数')
        ax9.set_ylabel('均值')
        ax9.grid(True, alpha=0.3)
        
        # 情绪时间线摘要
        ax10 = fig.add_subplot(gs[2, 2:])
        timeline = timeline_result.get('timeline', [])
        if timeline:
            dominant_emotions = [t['dominant_emotion'] for t in timeline]
            emotion_counts = {e: dominant_emotions.count(e) for e in set(dominant_emotions)}
            emotions = list(emotion_counts.keys())
            counts = list(emotion_counts.values())
            colors = [self.emotion_colors[e] for e in emotions]
            emotion_labels = [self.emotion_names_cn.get(e, e) for e in emotions]
            
            bars10 = ax10.bar(emotion_labels, counts, color=colors, alpha=0.8)
            ax10.set_title('情绪时间分布', fontsize=14, fontweight='bold')
            ax10.set_xlabel('情绪')
            ax10.set_ylabel('出现次数')
            ax10.grid(True, alpha=0.3, axis='y')
        
        # 性能信息
        ax11 = fig.add_subplot(gs[3, :])
        if performance_info:
            labels = []
            values = []
            if 'serial_time' in performance_info and performance_info['serial_time'] > 0:
                labels.append('串行处理时间(s)')
                values.append(performance_info['serial_time'])
            if 'parallel_time' in performance_info and performance_info['parallel_time'] > 0:
                labels.append('并行处理时间(s)')
                values.append(performance_info['parallel_time'])
            if 'speedup' in performance_info and performance_info['speedup'] > 0:
                labels.append('加速比')
                values.append(performance_info['speedup'])
            
            if labels:
                ax11.bar(labels, values, color=['#9E9E9E', '#607D8B', '#455A64'], alpha=0.8)
                ax11.set_title('性能指标', fontsize=14, fontweight='bold')
                ax11.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        if save_path is None:
            filename = os.path.basename(audio_file)
            save_path = os.path.join(self.output_dir, f"{os.path.splitext(filename)[0]}_comprehensive_report.png")
        
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"综合报告已保存至: {save_path}")
        return save_path
