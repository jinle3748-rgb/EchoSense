"""
情绪分析模块

功能：
- 基于声学特征的情绪识别
- 情绪强度分析
- 情绪变化趋势分析
- 多维度情绪评估

支持的情绪类别：
- 中性（Neutral）
- 快乐（Happy）
- 悲伤（Sad）
- 愤怒（Angry）
- 恐惧（Fear）
- 惊讶（Surprise）
- 厌恶（Disgust）
"""

import numpy as np
from collections import defaultdict
import warnings

warnings.filterwarnings('ignore')


class EmotionAnalyzer:
    """情绪分析器"""
    
    def __init__(self):
        """初始化情绪分析器"""
        self.emotion_categories = ['neutral', 'happy', 'sad', 'angry', 'fear', 'surprise', 'disgust']
        
        self.emotion_thresholds = {
            'happy': {
                'f0_high': True,
                'f0_variance_high': True,
                'energy_high': True,
                'spectral_centroid_high': True,
                'speaking_rate_high': True,
                'jitter_low': True
            },
            'sad': {
                'f0_low': True,
                'f0_variance_low': True,
                'energy_low': True,
                'spectral_centroid_low': True,
                'speaking_rate_low': True,
                'hnr_low': True
            },
            'angry': {
                'f0_high': True,
                'f0_variance_high': True,
                'energy_high': True,
                'spectral_centroid_high': True,
                'speaking_rate_high': True,
                'jitter_high': True,
                'shimmer_high': True
            },
            'fear': {
                'f0_high': True,
                'f0_variance_high': True,
                'energy_variable': True,
                'speaking_rate_variable': True,
                'jitter_high': True,
                'shimmer_high': True
            },
            'surprise': {
                'f0_high': True,
                'f0_range_high': True,
                'energy_high': True,
                'speaking_rate_high': True,
                'spectral_centroid_high': True
            },
            'disgust': {
                'f0_low': True,
                'energy_low': True,
                'spectral_centroid_low': True,
                'hnr_low': True
            },
            'neutral': {
                'f0_medium': True,
                'f0_variance_medium': True,
                'energy_medium': True,
                'speaking_rate_medium': True
            }
        }
        
        print("情绪分析器初始化完成")
        print(f"  支持情绪类别: {', '.join(self.emotion_categories)}")
    
    def analyze_emotion(self, features):
        """
        分析情绪
        
        Args:
            features: 声学特征字典
            
        Returns:
            dict: 情绪分析结果
        """
        emotion_scores = self._calculate_emotion_scores(features)
        
        dominant_emotion = max(emotion_scores, key=emotion_scores.get)
        confidence = emotion_scores[dominant_emotion]
        
        sorted_emotions = sorted(emotion_scores.items(), key=lambda x: x[1], reverse=True)
        secondary_emotion = sorted_emotions[1][0] if len(sorted_emotions) > 1 else 'neutral'
        secondary_confidence = sorted_emotions[1][1] if len(sorted_emotions) > 1 else 0
        
        intensity = self._calculate_emotion_intensity(features, dominant_emotion)
        
        result = {
            'dominant_emotion': dominant_emotion,
            'confidence': confidence,
            'secondary_emotion': secondary_emotion,
            'secondary_confidence': secondary_confidence,
            'intensity': intensity,
            'emotion_scores': emotion_scores,
            'emotion_indicators': self._get_emotion_indicators(features)
        }
        
        return result
    
    def _calculate_emotion_scores(self, features):
        """
        计算各情绪的得分
        
        Args:
            features: 声学特征
            
        Returns:
            dict: 各情绪得分
        """
        scores = defaultdict(float)
        
        f0_features = features.get('f0', {})
        energy_features = features.get('energy', {})
        spectral_features = features.get('spectral', {})
        quality_features = features.get('quality', {})
        temporal_features = features.get('temporal', {})
        
        mean_f0 = f0_features.get('mean_f0', 150)
        std_f0 = f0_features.get('std_f0', 20)
        range_f0 = f0_features.get('range_f0', 50)
        
        rms_mean = energy_features.get('rms_mean', 0.02)
        energy_entropy = energy_features.get('energy_entropy', 2)
        
        spectral_centroid = spectral_features.get('spectral_centroid_mean', 2000)
        spectral_bandwidth = spectral_features.get('spectral_bandwidth_mean', 1500)
        
        hnr = quality_features.get('hnr', 15)
        jitter = quality_features.get('jitter', 1)
        shimmer = quality_features.get('shimmer', 1)
        
        voiced_ratio = temporal_features.get('voiced_ratio', 0.5)
        num_segments = temporal_features.get('num_voiced_segments', 10)
        
        scores['happy'] = self._score_happy(mean_f0, std_f0, rms_mean, spectral_centroid, voiced_ratio, jitter)
        scores['sad'] = self._score_sad(mean_f0, std_f0, rms_mean, spectral_centroid, hnr, voiced_ratio)
        scores['angry'] = self._score_angry(mean_f0, std_f0, rms_mean, spectral_centroid, jitter, shimmer, voiced_ratio)
        scores['fear'] = self._score_fear(mean_f0, std_f0, rms_mean, jitter, shimmer, energy_entropy)
        scores['surprise'] = self._score_surprise(mean_f0, range_f0, rms_mean, spectral_centroid, voiced_ratio)
        scores['disgust'] = self._score_disgust(mean_f0, rms_mean, spectral_centroid, hnr)
        scores['neutral'] = self._score_neutral(mean_f0, std_f0, rms_mean, voiced_ratio)
        
        total = sum(scores.values())
        if total > 0:
            for emotion in scores:
                scores[emotion] = scores[emotion] / total
        
        return dict(scores)
    
    def _score_happy(self, f0, f0_std, energy, spectral_centroid, voiced_ratio, jitter):
        """计算快乐情绪得分"""
        score = 0
        
        if f0 > 200:
            score += 0.25
        elif f0 > 180:
            score += 0.15
        
        if f0_std > 30:
            score += 0.20
        elif f0_std > 20:
            score += 0.10
        
        if energy > 0.025:
            score += 0.20
        elif energy > 0.020:
            score += 0.10
        
        if spectral_centroid > 2500:
            score += 0.15
        
        if voiced_ratio > 0.6:
            score += 0.10
        
        if jitter < 1:
            score += 0.10
        
        return score
    
    def _score_sad(self, f0, f0_std, energy, spectral_centroid, hnr, voiced_ratio):
        """计算悲伤情绪得分"""
        score = 0
        
        if f0 < 150:
            score += 0.30
        elif f0 < 180:
            score += 0.15
        
        if f0_std < 15:
            score += 0.20
        elif f0_std < 25:
            score += 0.10
        
        if energy < 0.015:
            score += 0.25
        elif energy < 0.020:
            score += 0.15
        
        if spectral_centroid < 2000:
            score += 0.15
        
        if hnr < 15:
            score += 0.10
        
        if voiced_ratio < 0.4:
            score += 0.10
        
        return score
    
    def _score_angry(self, f0, f0_std, energy, spectral_centroid, jitter, shimmer, voiced_ratio):
        """计算愤怒情绪得分"""
        score = 0
        
        if f0 > 220:
            score += 0.25
        elif f0 > 200:
            score += 0.15
        
        if f0_std > 40:
            score += 0.20
        elif f0_std > 30:
            score += 0.10
        
        if energy > 0.03:
            score += 0.25
        elif energy > 0.025:
            score += 0.15
        
        if spectral_centroid > 3000:
            score += 0.15
        
        if jitter > 2:
            score += 0.10
        
        if shimmer > 2:
            score += 0.10
        
        if voiced_ratio > 0.7:
            score += 0.05
        
        return score
    
    def _score_fear(self, f0, f0_std, energy, jitter, shimmer, energy_entropy):
        """计算恐惧情绪得分"""
        score = 0
        
        if f0 > 200:
            score += 0.20
        
        if f0_std > 35:
            score += 0.25
        
        if jitter > 2:
            score += 0.20
        
        if shimmer > 2:
            score += 0.15
        
        if energy_entropy > 3:
            score += 0.20
        
        return score
    
    def _score_surprise(self, f0, f0_range, energy, spectral_centroid, voiced_ratio):
        """计算惊讶情绪得分"""
        score = 0
        
        if f0 > 200:
            score += 0.25
        
        if f0_range > 80:
            score += 0.30
        elif f0_range > 50:
            score += 0.15
        
        if energy > 0.025:
            score += 0.20
        
        if spectral_centroid > 2500:
            score += 0.15
        
        if voiced_ratio > 0.5:
            score += 0.10
        
        return score
    
    def _score_disgust(self, f0, energy, spectral_centroid, hnr):
        """计算厌恶情绪得分"""
        score = 0
        
        if f0 < 150:
            score += 0.30
        
        if energy < 0.015:
            score += 0.25
        
        if spectral_centroid < 1800:
            score += 0.25
        
        if hnr < 12:
            score += 0.20
        
        return score
    
    def _score_neutral(self, f0, f0_std, energy, voiced_ratio):
        """计算中性情绪得分"""
        score = 0
        
        if 150 <= f0 <= 200:
            score += 0.30
        
        if 15 <= f0_std <= 30:
            score += 0.25
        
        if 0.015 <= energy <= 0.025:
            score += 0.25
        
        if 0.4 <= voiced_ratio <= 0.6:
            score += 0.20
        
        return score
    
    def _calculate_emotion_intensity(self, features, emotion):
        """
        计算情绪强度
        
        Args:
            features: 声学特征
            emotion: 情绪类型
            
        Returns:
            float: 情绪强度（0-1）
        """
        f0_features = features.get('f0', {})
        energy_features = features.get('energy', {})
        
        f0_range = f0_features.get('range_f0', 0)
        energy_range = energy_features.get('dynamic_range', 0)
        
        intensity = 0
        
        if emotion in ['happy', 'angry', 'surprise']:
            intensity = (f0_range / 100 + energy_range * 10) / 2
        elif emotion in ['sad', 'disgust']:
            intensity = (1 - f0_range / 100 + 1 - energy_range * 10) / 2
        elif emotion == 'fear':
            intensity = (f0_features.get('std_f0', 0) / 50) * 0.5 + 0.5
        else:
            intensity = 0.3
        
        return min(max(intensity, 0), 1)
    
    def _get_emotion_indicators(self, features):
        """
        获取情绪指示器
        
        Args:
            features: 声学特征
            
        Returns:
            dict: 情绪指示器
        """
        indicators = {}
        
        f0_features = features.get('f0', {})
        energy_features = features.get('energy', {})
        spectral_features = features.get('spectral', {})
        quality_features = features.get('quality', {})
        
        mean_f0 = f0_features.get('mean_f0', 0)
        std_f0 = f0_features.get('std_f0', 0)
        
        indicators['pitch_level'] = 'high' if mean_f0 > 200 else 'low' if mean_f0 < 150 else 'medium'
        indicators['pitch_variability'] = 'high' if std_f0 > 30 else 'low' if std_f0 < 15 else 'medium'
        
        rms_mean = energy_features.get('rms_mean', 0)
        indicators['energy_level'] = 'high' if rms_mean > 0.025 else 'low' if rms_mean < 0.015 else 'medium'
        
        spectral_centroid = spectral_features.get('spectral_centroid_mean', 0)
        indicators['brightness'] = 'bright' if spectral_centroid > 2500 else 'dark' if spectral_centroid < 2000 else 'neutral'
        
        jitter = quality_features.get('jitter', 0)
        shimmer = quality_features.get('shimmer', 0)
        indicators['voice_quality'] = 'stable' if jitter < 1 and shimmer < 1 else 'unstable'
        
        hnr = quality_features.get('hnr', 0)
        indicators['harmonic_content'] = 'rich' if hnr > 20 else 'poor' if hnr < 15 else 'moderate'
        
        return indicators
    
    def analyze_emotion_timeline(self, features_timeline):
        """
        分析情绪时间线
        
        Args:
            features_timeline: 特征时间线列表
            
        Returns:
            dict: 情绪时间线分析结果
        """
        emotions_timeline = []
        
        for features in features_timeline:
            emotion_result = self.analyze_emotion(features)
            emotions_timeline.append(emotion_result)
        
        emotion_counts = defaultdict(int)
        for result in emotions_timeline:
            emotion_counts[result['dominant_emotion']] += 1
        
        total = len(emotions_timeline)
        emotion_distribution = {
            emotion: count / total for emotion, count in emotion_counts.items()
        }
        
        if len(emotions_timeline) > 1:
            emotion_changes = []
            for i in range(1, len(emotions_timeline)):
                prev = emotions_timeline[i-1]['dominant_emotion']
                curr = emotions_timeline[i]['dominant_emotion']
                if prev != curr:
                    emotion_changes.append({
                        'from': prev,
                        'to': curr,
                        'position': i
                    })
        else:
            emotion_changes = []
        
        avg_intensity = np.mean([r['intensity'] for r in emotions_timeline])
        avg_confidence = np.mean([r['confidence'] for r in emotions_timeline])
        
        return {
            'timeline': emotions_timeline,
            'distribution': dict(emotion_distribution),
            'changes': emotion_changes,
            'num_changes': len(emotion_changes),
            'avg_intensity': avg_intensity,
            'avg_confidence': avg_confidence,
            'dominant_overall': max(emotion_distribution, key=emotion_distribution.get)
        }
    
    def get_emotion_description(self, emotion, intensity, confidence):
        """
        获取情绪描述
        
        Args:
            emotion: 情绪类型
            intensity: 情绪强度
            confidence: 置信度
            
        Returns:
            str: 情绪描述
        """
        emotion_names_cn = {
            'neutral': '中性',
            'happy': '快乐',
            'sad': '悲伤',
            'angry': '愤怒',
            'fear': '恐惧',
            'surprise': '惊讶',
            'disgust': '厌恶'
        }
        
        intensity_desc = '强烈' if intensity > 0.7 else '中等' if intensity > 0.4 else '轻微'
        confidence_desc = '高' if confidence > 0.7 else '中等' if confidence > 0.5 else '低'
        
        emotion_cn = emotion_names_cn.get(emotion, emotion)
        
        return f"{intensity_desc}{emotion_cn}情绪（置信度：{confidence_desc}）"
