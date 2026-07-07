#!/usr/bin/env python3
"""
高级性别识别模块

功能：
- 基于多种声学特征的性别识别
- 支持多维度特征分析
- 提供详细的分析报告
"""

import numpy as np
import librosa
import json

class AdvancedGenderAnalyzer:
    """高级性别分析器"""
    
    def __init__(self):
        """初始化高级性别分析器"""
        self.f0_male_range = (80, 165)
        self.f0_female_range = (165, 300)
        
        print("Advanced Gender Analyzer initialized")
    
    def extract_features(self, audio_path):
        """
        提取音频特征
        
        Args:
            audio_path: 音频文件路径
            
        Returns:
            dict: 特征字典
        """
        y, sr = librosa.load(audio_path, sr=16000)
        
        features = {}
        
        # 基频特征
        f0, _, _ = librosa.pyin(y, fmin=60, fmax=400, sr=sr)
        valid_f0 = f0[~np.isnan(f0)]
        if len(valid_f0) > 0:
            features['f0_mean'] = float(np.mean(valid_f0))
            features['f0_std'] = float(np.std(valid_f0))
            features['f0_min'] = float(np.min(valid_f0))
            features['f0_max'] = float(np.max(valid_f0))
        else:
            features['f0_mean'] = 0
            features['f0_std'] = 0
            features['f0_min'] = 0
            features['f0_max'] = 0
        
        # 能量特征
        rms = librosa.feature.rms(y=y)
        features['rms_mean'] = float(np.mean(rms))
        features['rms_std'] = float(np.std(rms))
        
        # 频谱特征
        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
        features['spectral_centroid_mean'] = float(np.mean(spectral_centroid))
        
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
        features['spectral_bandwidth_mean'] = float(np.mean(spectral_bandwidth))
        
        # MFCC特征
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        for i in range(13):
            features[f'mfcc_{i}_mean'] = float(np.mean(mfccs[i]))
        
        return features
    
    def analyze(self, audio_path):
        """
        分析音频文件的性别
        
        Args:
            audio_path: 音频文件路径
            
        Returns:
            dict: 分析结果
        """
        try:
            features = self.extract_features(audio_path)
            f0_mean = features['f0_mean']
            
            # 基于基频判断性别
            if f0_mean < self.f0_male_range[1]:
                gender = 'male'
                confidence = self._calculate_confidence(f0_mean, is_male=True)
            elif f0_mean > self.f0_female_range[0]:
                gender = 'female'
                confidence = self._calculate_confidence(f0_mean, is_male=False)
            else:
                gender = 'neutral'
                confidence = 0.55
            
            result = {
                'gender': gender,
                'confidence': float(min(confidence, 1.0)),
                'features': features,
                'analysis_method': 'advanced'
            }
            
            return result
            
        except Exception as e:
            return {
                'gender': 'error',
                'confidence': 0.0,
                'error': str(e)
            }
    
    def _calculate_confidence(self, f0_mean, is_male=True):
        """计算识别置信度"""
        if is_male:
            if f0_mean < 100:
                return 0.95
            elif f0_mean < 130:
                return 0.85
            elif f0_mean < 150:
                return 0.70
            else:
                return 0.55
        else:
            if f0_mean > 250:
                return 0.95
            elif f0_mean > 200:
                return 0.85
            elif f0_mean > 180:
                return 0.70
            else:
                return 0.55
    
    def save_result(self, result, output_path):
        """
        保存分析结果
        
        Args:
            result: 分析结果
            output_path: 输出文件路径
        """
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
