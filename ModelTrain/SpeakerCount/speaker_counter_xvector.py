"""
基于XVector的说话人计数模块

使用项目中已有的XVector模型进行说话人计数
无需网络，完全离线使用

算法流程：
1. 音频分段（使用能量检测静音）
2. 提取每段的XVector嵌入
3. 使用聚类算法（DBSCAN/层次聚类）分组
4. 统计聚类数量作为说话人数量
"""

import os
import pickle
import warnings

import numpy as np
from sklearn.cluster import DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score

warnings.filterwarnings('ignore')


class XVectorSpeakerCounter:
    """基于XVector的说话人计数器"""
    
    def __init__(self, model_dir: str = "models/XVector"):
        """
        初始化XVector说话人计数器
        
        Args:
            model_dir: XVector模型目录路径
        """
        self.model_dir = model_dir
        self.xvector_model = None
        self.speaker_to_id = None
        self._load_models()
    
    def _load_models(self):
        """加载XVector模型"""
        try:
            from tensorflow.keras.models import load_model
            
            # 加载XVector模型
            model_path = os.path.join(self.model_dir, "xvector_model.h5")
            embedding_path = os.path.join(self.model_dir, "xvector_embedding.h5")
            
            if os.path.exists(model_path):
                self.xvector_model = load_model(model_path)
                print(f"✓ 加载XVector模型: {model_path}")
            elif os.path.exists(embedding_path):
                self.xvector_model = load_model(embedding_path)
                print(f"✓ 加载XVector嵌入模型: {embedding_path}")
            else:
                raise FileNotFoundError(f"未找到XVector模型文件")
            
            # 加载speaker_to_id映射（可选）
            speaker_to_id_path = os.path.join(self.model_dir, "speaker_to_id.pkl")
            if os.path.exists(speaker_to_id_path):
                with open(speaker_to_id_path, 'rb') as f:
                    self.speaker_to_id = pickle.load(f)
                print(f"✓ 加载speaker_to_id映射")
            
            print("XVector说话人计数器初始化完成")
            
        except ImportError:
            raise ImportError("请安装 tensorflow: pip install tensorflow")
        except Exception as e:
            raise RuntimeError(f"加载模型失败: {e}")
    
    def _extract_audio_segments(self, audio_path: str, segment_duration: float = 1.0):
        """
        提取音频片段
        
        Args:
            audio_path: 音频文件路径
            segment_duration: 每个片段的时长（秒）
            
        Returns:
            list: 音频片段列表（numpy数组）
            list: 片段起始时间列表
        """
        import librosa
        
        y, sr = librosa.load(audio_path, sr=16000)
        
        samples_per_segment = int(segment_duration * sr)
        num_segments = len(y) // samples_per_segment
        
        segments = []
        timestamps = []
        
        for i in range(num_segments):
            start = i * samples_per_segment
            end = start + samples_per_segment
            segment = y[start:end]
            
            # 跳过静音片段
            rms = librosa.feature.rms(y=segment).mean()
            if rms > 0.001:
                segments.append(segment)
                timestamps.append(i * segment_duration)
        
        return segments, timestamps
    
    def _extract_embedding(self, audio_segment: np.ndarray, sr: int = 16000):
        """
        提取单段音频的XVector嵌入
        
        Args:
            audio_segment: 音频片段（numpy数组）
            sr: 采样率
            
        Returns:
            np.ndarray: XVector嵌入向量
        """
        import librosa
        
        # 提取MFCC特征
        mfcc = librosa.feature.mfcc(
            y=audio_segment,
            sr=sr,
            n_mfcc=20,
            n_fft=512,
            hop_length=160
        )
        
        # 归一化
        mfcc = (mfcc - np.mean(mfcc, axis=1, keepdims=True)) / (np.std(mfcc, axis=1, keepdims=True) + 1e-8)
        
        # 转置为 (time_steps, n_mfcc)
        mfcc = mfcc.T
        
        # 添加batch维度: (1, time_steps, n_mfcc)
        mfcc = np.expand_dims(mfcc, axis=0)
        
        # 提取嵌入
        embedding = self.xvector_model.predict(mfcc, verbose=0)
        
        return embedding.flatten()
    
    def _cluster_embeddings(self, embeddings: np.ndarray, method: str = 'agglomerative'):
        """
        对嵌入向量进行聚类
        
        Args:
            embeddings: 嵌入向量矩阵 (n_samples, n_features)
            method: 聚类方法 ('dbscan', 'agglomerative')
            
        Returns:
            tuple: (聚类标签, 说话人数量)
        """
        if len(embeddings) < 2:
            return np.array([0]), 1
        
        if method == 'dbscan':
            # DBSCAN聚类
            dbscan = DBSCAN(eps=0.5, min_samples=2)
            labels = dbscan.fit_predict(embeddings)
            
            # 计算噪声点比例
            noise_ratio = np.sum(labels == -1) / len(labels)
            if noise_ratio > 0.5:
                # 如果太多噪声点，使用层次聚类
                method = 'agglomerative'
        
        if method == 'agglomerative' or method == 'dbscan':
            # 层次聚类 - 尝试找到最佳聚类数
            best_n_clusters = 1
            best_score = -1
            
            max_clusters = min(10, len(embeddings) // 2 + 1)
            
            for n_clusters in range(2, max_clusters + 1):
                clustering = AgglomerativeClustering(n_clusters=n_clusters)
                labels = clustering.fit_predict(embeddings)
                
                try:
                    score = silhouette_score(embeddings, labels)
                    if score > best_score:
                        best_score = score
                        best_n_clusters = n_clusters
                except:
                    continue
            
            # 使用最佳聚类数
            clustering = AgglomerativeClustering(n_clusters=best_n_clusters)
            labels = clustering.fit_predict(embeddings)
        
        num_speakers = len(set(labels))
        
        return labels, num_speakers
    
    def count_speakers(self, audio_path: str, segment_duration: float = 1.0) -> dict:
        """
        统计音频中的说话人数量
        
        Args:
            audio_path: 音频文件路径
            segment_duration: 每个分析片段的时长（秒）
            
        Returns:
            dict: 分析结果
        """
        # 提取音频片段
        segments, timestamps = self._extract_audio_segments(audio_path, segment_duration)
        
        if len(segments) == 0:
            return {
                'num_speakers': 0,
                'speaker_labels': [],
                'audio_duration': 0,
                'speaker_durations': {},
                'speaker_timeline': [],
                'warning': '未检测到语音内容'
            }
        
        # 提取嵌入向量
        embeddings = []
        for segment in segments:
            embedding = self._extract_embedding(segment)
            embeddings.append(embedding)
        
        embeddings = np.array(embeddings)
        
        # 聚类
        labels, num_speakers = self._cluster_embeddings(embeddings)
        
        # 计算各说话人时长
        speaker_durations = {}
        for label, timestamp in zip(labels, timestamps):
            speaker = f"speaker_{label}"
            if speaker not in speaker_durations:
                speaker_durations[speaker] = 0.0
            speaker_durations[speaker] += segment_duration
        
        # 创建时间线
        speaker_timeline = []
        for label, timestamp in zip(labels, timestamps):
            speaker_timeline.append({
                'start': timestamp,
                'end': timestamp + segment_duration,
                'duration': segment_duration,
                'speaker': f"speaker_{label}"
            })
        
        # 计算音频总时长
        audio_duration = timestamps[-1] + segment_duration if timestamps else 0
        
        result = {
            'num_speakers': num_speakers,
            'speaker_labels': [f"speaker_{i}" for i in range(num_speakers)],
            'audio_duration': audio_duration,
            'speaker_durations': speaker_durations,
            'speaker_timeline': speaker_timeline,
            'method': 'xvector'
        }
        
        return result
    
    def analyze_speaker_activity(self, audio_path: str) -> dict:
        """
        分析说话人活动
        
        Args:
            audio_path: 音频文件路径
            
        Returns:
            dict: 活动分析结果
        """
        result = self.count_speakers(audio_path)
        
        if result.get('warning'):
            return result
        
        total_duration = result['audio_duration']
        speaker_durations = result['speaker_durations']
        
        speaker_ratios = {
            speaker: duration / total_duration 
            for speaker, duration in speaker_durations.items()
        }
        
        total_speech_time = sum(speaker_durations.values())
        silence_time = total_duration - total_speech_time
        silence_ratio = silence_time / total_duration if total_duration > 0 else 0
        
        main_speaker = max(speaker_durations, key=speaker_durations.get) if speaker_durations else None
        
        timeline = result['speaker_timeline']
        turn_takes = 0
        for i in range(1, len(timeline)):
            if timeline[i]['speaker'] != timeline[i-1]['speaker']:
                turn_takes += 1
        
        analysis = {
            'num_speakers': result['num_speakers'],
            'total_duration': total_duration,
            'total_speech_time': total_speech_time,
            'silence_time': silence_time,
            'silence_ratio': silence_ratio,
            'speaker_durations': speaker_durations,
            'speaker_ratios': speaker_ratios,
            'main_speaker': main_speaker,
            'main_speaker_ratio': speaker_ratios.get(main_speaker, 0),
            'num_turn_takes': turn_takes,
            'avg_turn_duration': total_speech_time / (turn_takes + 1) if turn_takes > 0 else total_speech_time,
            'method': 'xvector'
        }
        
        return analysis
    
    def print_summary(self, result: dict):
        """打印分析结果摘要"""
        print("\n" + "="*50)
        print("说话人分析结果 (XVector)")
        print("="*50)
        
        if 'warning' in result:
            print(f"警告: {result['warning']}")
            print("="*50)
            return
        
        print(f"检测到的说话人数量: {result['num_speakers']}")
        print(f"音频总时长: {result['audio_duration']:.2f} 秒")
        
        if 'speaker_durations' in result:
            print("\n各说话人说话时长:")
            for speaker, duration in sorted(result['speaker_durations'].items()):
                if 'speaker_ratios' in result:
                    ratio = result['speaker_ratios'][speaker]
                    print(f"  {speaker}: {duration:.2f} 秒 ({ratio*100:.1f}%)")
                else:
                    print(f"  {speaker}: {duration:.2f} 秒")
        
        if 'num_turn_takes' in result:
            print(f"\n说话人交替次数: {result['num_turn_takes']}")
            print(f"主要说话人: {result['main_speaker']} ({result['main_speaker_ratio']*100:.1f}%)")
            print(f"静音比例: {result['silence_ratio']*100:.1f}%")
        
        print("="*50)