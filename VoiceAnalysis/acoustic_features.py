"""
声学特征提取模块

功能：
- 基频（F0）特征提取
- 共振峰（Formants）特征提取
- 频谱特征提取
- 能量特征提取
- MFCC特征提取
- 语音质量特征提取
"""

import numpy as np
import librosa
from scipy import signal
from scipy.signal import lfilter, find_peaks
import warnings

warnings.filterwarnings('ignore')


class AcousticFeatureExtractor:
    """声学特征提取器"""
    
    def __init__(self, sr=None, n_fft=2048, hop_length=512, window_length=0.025):
        """
        初始化声学特征提取器
        
        Args:
            sr: 采样率，None表示自动检测
            n_fft: FFT窗口大小
            hop_length: hop长度
            window_length: 窗口长度（秒）
        """
        self.sr = sr
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.window_length = window_length
        self.window_samples = int(window_length * (sr if sr else 16000))
        
        print("声学特征提取器初始化完成")
        print(f"  FFT窗口: {n_fft}")
        print(f"  Hop长度: {hop_length}")
        print(f"  分析窗口: {window_length}秒")
    
    def extract_all_features(self, audio, sr):
        """
        提取所有声学特征
        
        Args:
            audio: 音频信号
            sr: 采样率
            
        Returns:
            dict: 包含所有特征的字典
        """
        features = {}
        
        features['f0'] = self.extract_f0_features(audio, sr)
        features['formants'] = self.extract_formant_features(audio, sr)
        features['spectral'] = self.extract_spectral_features(audio, sr)
        features['energy'] = self.extract_energy_features(audio, sr)
        features['mfcc'] = self.extract_mfcc_features(audio, sr)
        features['quality'] = self.extract_quality_features(audio, sr)
        features['temporal'] = self.extract_temporal_features(audio, sr)
        
        return features
    
    def extract_f0_features(self, audio, sr):
        """
        提取基频特征
        
        Args:
            audio: 音频信号
            sr: 采样率
            
        Returns:
            dict: 基频特征
        """
        f0_features = {}
        
        try:
            f0, voiced_flags, voiced_probs = librosa.pyin(
                audio, 
                fmin=60, 
                fmax=400, 
                sr=sr,
                frame_length=self.n_fft,
                hop_length=self.hop_length
            )
            
            valid_f0 = f0[~np.isnan(f0)]
            
            if len(valid_f0) > 0:
                f0_features['mean_f0'] = float(np.mean(valid_f0))
                f0_features['median_f0'] = float(np.median(valid_f0))
                f0_features['std_f0'] = float(np.std(valid_f0))
                f0_features['min_f0'] = float(np.min(valid_f0))
                f0_features['max_f0'] = float(np.max(valid_f0))
                f0_features['range_f0'] = f0_features['max_f0'] - f0_features['min_f0']
                f0_features['variance_f0'] = float(np.var(valid_f0))
                f0_features['q1_f0'] = float(np.percentile(valid_f0, 25))
                f0_features['q3_f0'] = float(np.percentile(valid_f0, 75))
                f0_features['iqr_f0'] = f0_features['q3_f0'] - f0_features['q1_f0']
                
                f0_diff = np.diff(valid_f0)
                f0_features['mean_diff_f0'] = float(np.mean(np.abs(f0_diff)))
                f0_features['std_diff_f0'] = float(np.std(f0_diff))
                
                f0_features['voiced_ratio'] = float(np.sum(voiced_flags) / len(voiced_flags))
                f0_features['mean_voiced_prob'] = float(np.mean(voiced_probs))
                
                f0_centered = valid_f0 - np.mean(valid_f0)
                f0_features['skewness_f0'] = float(
                    np.mean(f0_centered ** 3) / (np.std(valid_f0) ** 3 + 1e-10)
                )
                f0_features['kurtosis_f0'] = float(
                    np.mean(f0_centered ** 4) / (np.std(valid_f0) ** 4 + 1e-10) - 3
                )
            else:
                f0_features = self._get_default_f0_features()
                
        except Exception as e:
            print(f"基频提取错误: {e}")
            f0_features = self._get_default_f0_features()
        
        return f0_features
    
    def _get_default_f0_features(self):
        """返回默认的基频特征"""
        return {
            'mean_f0': 0, 'median_f0': 0, 'std_f0': 0,
            'min_f0': 0, 'max_f0': 0, 'range_f0': 0,
            'variance_f0': 0, 'q1_f0': 0, 'q3_f0': 0,
            'iqr_f0': 0, 'mean_diff_f0': 0, 'std_diff_f0': 0,
            'voiced_ratio': 0, 'mean_voiced_prob': 0,
            'skewness_f0': 0, 'kurtosis_f0': 0
        }
    
    def extract_formant_features(self, audio, sr):
        """
        提取共振峰特征
        
        Args:
            audio: 音频信号
            sr: 采样率
            
        Returns:
            dict: 共振峰特征
        """
        formant_features = {}
        
        try:
            lpc_order = int(2 + sr / 1000)
            lpc_coeffs = librosa.lpc(audio, order=lpc_order)
            
            roots = np.roots(lpc_coeffs)
            roots = roots[np.imag(roots) >= 0]
            
            angz = np.arctan2(np.imag(roots), np.real(roots))
            formants = angz * (sr / (2 * np.pi))
            bandwidths = -0.5 * (angz / (2 * np.pi)) * sr
            
            formants = formants[formants > 0]
            formants = np.sort(formants)
            
            if len(formants) >= 3:
                formant_features['f1'] = float(formants[0])
                formant_features['f2'] = float(formants[1])
                formant_features['f3'] = float(formants[2])
                formant_features['f4'] = float(formants[3]) if len(formants) > 3 else 0
                
                formant_features['f2_f1_ratio'] = formant_features['f2'] / (formant_features['f1'] + 1e-10)
                formant_features['f3_f1_ratio'] = formant_features['f3'] / (formant_features['f1'] + 1e-10)
                formant_features['f2_f1_diff'] = formant_features['f2'] - formant_features['f1']
                formant_features['f3_f1_diff'] = formant_features['f3'] - formant_features['f1']
                
                formant_features['formant_dispersion'] = float(np.std(formants[:3]))
                formant_features['formant_avg'] = float(np.mean(formants[:3]))
                
                if len(bandwidths) >= 3:
                    valid_bw = bandwidths[:3][bandwidths[:3] > 0]
                    if len(valid_bw) > 0:
                        formant_features['b1'] = float(valid_bw[0]) if len(valid_bw) > 0 else 0
                        formant_features['b2'] = float(valid_bw[1]) if len(valid_bw) > 1 else 0
                        formant_features['b3'] = float(valid_bw[2]) if len(valid_bw) > 2 else 0
            else:
                formant_features = self._get_default_formant_features()
                
        except Exception as e:
            print(f"共振峰提取错误: {e}")
            formant_features = self._get_default_formant_features()
        
        return formant_features
    
    def _get_default_formant_features(self):
        """返回默认的共振峰特征"""
        return {
            'f1': 0, 'f2': 0, 'f3': 0, 'f4': 0,
            'f2_f1_ratio': 0, 'f3_f1_ratio': 0,
            'f2_f1_diff': 0, 'f3_f1_diff': 0,
            'formant_dispersion': 0, 'formant_avg': 0,
            'b1': 0, 'b2': 0, 'b3': 0
        }
    
    def extract_spectral_features(self, audio, sr):
        """
        提取频谱特征
        
        Args:
            audio: 音频信号
            sr: 采样率
            
        Returns:
            dict: 频谱特征
        """
        spectral_features = {}
        
        try:
            stft = np.abs(librosa.stft(audio, n_fft=self.n_fft, hop_length=self.hop_length))
            
            spectral_centroid = librosa.feature.spectral_centroid(S=stft, sr=sr)[0]
            spectral_features['spectral_centroid_mean'] = float(np.mean(spectral_centroid))
            spectral_features['spectral_centroid_std'] = float(np.std(spectral_centroid))
            spectral_features['spectral_centroid_max'] = float(np.max(spectral_centroid))
            
            spectral_bandwidth = librosa.feature.spectral_bandwidth(S=stft, sr=sr)[0]
            spectral_features['spectral_bandwidth_mean'] = float(np.mean(spectral_bandwidth))
            spectral_features['spectral_bandwidth_std'] = float(np.std(spectral_bandwidth))
            
            spectral_rolloff_85 = librosa.feature.spectral_rolloff(S=stft, sr=sr, roll_percent=0.85)[0]
            spectral_rolloff_95 = librosa.feature.spectral_rolloff(S=stft, sr=sr, roll_percent=0.95)[0]
            spectral_features['spectral_rolloff_85_mean'] = float(np.mean(spectral_rolloff_85))
            spectral_features['spectral_rolloff_95_mean'] = float(np.mean(spectral_rolloff_95))
            
            try:
                spectral_contrast = librosa.feature.spectral_contrast(S=stft, sr=sr)
                spectral_features['spectral_contrast_mean'] = float(np.mean(spectral_contrast))
                spectral_features['spectral_contrast_std'] = float(np.std(spectral_contrast))
            except:
                spectral_features['spectral_contrast_mean'] = 0
                spectral_features['spectral_contrast_std'] = 0
            
            spectral_flatness = librosa.feature.spectral_flatness(S=stft)[0]
            spectral_features['spectral_flatness_mean'] = float(np.mean(spectral_flatness))
            spectral_features['spectral_flatness_std'] = float(np.std(spectral_flatness))
            
            stft_power = stft ** 2
            stft_normalized = stft_power / (np.sum(stft_power) + 1e-10)
            spectral_features['spectral_entropy'] = float(
                -np.sum(stft_normalized * np.log2(stft_normalized + 1e-10))
            )
            
            spectral_features['spectral_flux'] = float(
                np.mean(np.sum(np.diff(stft, axis=1) ** 2, axis=0))
            )
            
        except Exception as e:
            print(f"频谱特征提取错误: {e}")
            spectral_features = self._get_default_spectral_features()
        
        return spectral_features
    
    def _get_default_spectral_features(self):
        """返回默认的频谱特征"""
        return {
            'spectral_centroid_mean': 0, 'spectral_centroid_std': 0, 'spectral_centroid_max': 0,
            'spectral_bandwidth_mean': 0, 'spectral_bandwidth_std': 0,
            'spectral_rolloff_85_mean': 0, 'spectral_rolloff_95_mean': 0,
            'spectral_contrast_mean': 0, 'spectral_contrast_std': 0,
            'spectral_flatness_mean': 0, 'spectral_flatness_std': 0,
            'spectral_entropy': 0, 'spectral_flux': 0
        }
    
    def extract_energy_features(self, audio, sr):
        """
        提取能量特征
        
        Args:
            audio: 音频信号
            sr: 采样率
            
        Returns:
            dict: 能量特征
        """
        energy_features = {}
        
        try:
            rms = librosa.feature.rms(y=audio, hop_length=self.hop_length)[0]
            energy_features['rms_mean'] = float(np.mean(rms))
            energy_features['rms_std'] = float(np.std(rms))
            energy_features['rms_max'] = float(np.max(rms))
            energy_features['rms_min'] = float(np.min(rms))
            
            energy = audio ** 2
            energy_features['total_energy'] = float(np.sum(energy))
            energy_features['mean_energy'] = float(np.mean(energy))
            
            energy_normalized = rms / (np.sum(rms) + 1e-10)
            energy_normalized = energy_normalized[energy_normalized > 0]
            energy_features['energy_entropy'] = float(
                -np.sum(energy_normalized * np.log2(energy_normalized + 1e-10))
            )
            
            energy_features['dynamic_range'] = float(
                np.max(rms) - np.min(rms)
            )
            
            threshold = np.mean(rms) + np.std(rms)
            energy_features['energy_variability'] = float(
                np.sum(rms > threshold) / len(rms)
            )
            
        except Exception as e:
            print(f"能量特征提取错误: {e}")
            energy_features = self._get_default_energy_features()
        
        return energy_features
    
    def _get_default_energy_features(self):
        """返回默认的能量特征"""
        return {
            'rms_mean': 0, 'rms_std': 0, 'rms_max': 0, 'rms_min': 0,
            'total_energy': 0, 'mean_energy': 0, 'energy_entropy': 0,
            'dynamic_range': 0, 'energy_variability': 0
        }
    
    def extract_mfcc_features(self, audio, sr):
        """
        提取MFCC特征
        
        Args:
            audio: 音频信号
            sr: 采样率
            
        Returns:
            dict: MFCC特征
        """
        mfcc_features = {}
        
        try:
            n_mfcc = 13
            mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc, 
                                         n_fft=self.n_fft, hop_length=self.hop_length)
            
            for i in range(n_mfcc):
                mfcc_features[f'mfcc_{i}_mean'] = float(np.mean(mfccs[i]))
                mfcc_features[f'mfcc_{i}_std'] = float(np.std(mfccs[i]))
            
            mfcc_delta = librosa.feature.delta(mfccs)
            mfcc_delta2 = librosa.feature.delta(mfccs, order=2)
            
            mfcc_features['mfcc_delta_mean'] = float(np.mean(mfcc_delta))
            mfcc_features['mfcc_delta_std'] = float(np.std(mfcc_delta))
            mfcc_features['mfcc_delta2_mean'] = float(np.mean(mfcc_delta2))
            mfcc_features['mfcc_delta2_std'] = float(np.std(mfcc_delta2))
            
            mfcc_features['mfcc_skewness'] = float(
                np.mean([np.mean((mfccs[i] - np.mean(mfccs[i])) ** 3) / 
                        (np.std(mfccs[i]) ** 3 + 1e-10) for i in range(n_mfcc)])
            )
            
        except Exception as e:
            print(f"MFCC特征提取错误: {e}")
            mfcc_features = self._get_default_mfcc_features()
        
        return mfcc_features
    
    def _get_default_mfcc_features(self):
        """返回默认的MFCC特征"""
        mfcc_features = {}
        for i in range(13):
            mfcc_features[f'mfcc_{i}_mean'] = 0
            mfcc_features[f'mfcc_{i}_std'] = 0
        mfcc_features['mfcc_delta_mean'] = 0
        mfcc_features['mfcc_delta_std'] = 0
        mfcc_features['mfcc_delta2_mean'] = 0
        mfcc_features['mfcc_delta2_std'] = 0
        mfcc_features['mfcc_skewness'] = 0
        return mfcc_features
    
    def extract_quality_features(self, audio, sr):
        """
        提取语音质量特征
        
        Args:
            audio: 音频信号
            sr: 采样率
            
        Returns:
            dict: 语音质量特征
        """
        quality_features = {}
        
        try:
            harmonic, percussive = librosa.effects.hpss(audio)
            
            harmonic_energy = np.sum(harmonic ** 2)
            percussive_energy = np.sum(percussive ** 2)
            
            if percussive_energy > 1e-10:
                quality_features['hnr'] = float(
                    10 * np.log10(harmonic_energy / percussive_energy)
                )
            else:
                quality_features['hnr'] = 50
            
            quality_features['harmonic_ratio'] = float(
                harmonic_energy / (harmonic_energy + percussive_energy + 1e-10)
            )
            
            zcr = librosa.feature.zero_crossing_rate(audio, hop_length=self.hop_length)[0]
            quality_features['zcr_mean'] = float(np.mean(zcr))
            quality_features['zcr_std'] = float(np.std(zcr))
            quality_features['zcr_max'] = float(np.max(zcr))
            
            try:
                f0, _, _ = librosa.pyin(audio, fmin=60, fmax=400, sr=sr)
                valid_f0 = f0[~np.isnan(f0)]
                
                if len(valid_f0) > 10:
                    f0_diff = np.diff(valid_f0)
                    jitter = np.mean(np.abs(f0_diff)) / (np.mean(valid_f0) + 1e-10)
                    quality_features['jitter'] = float(jitter * 100)
                    
                    amplitude_envelope = librosa.feature.rms(y=audio, hop_length=self.hop_length)[0]
                    amp_diff = np.diff(amplitude_envelope)
                    shimmer = np.mean(np.abs(amp_diff)) / (np.mean(amplitude_envelope) + 1e-10)
                    quality_features['shimmer'] = float(shimmer * 100)
                else:
                    quality_features['jitter'] = 0
                    quality_features['shimmer'] = 0
            except:
                quality_features['jitter'] = 0
                quality_features['shimmer'] = 0
            
        except Exception as e:
            print(f"语音质量特征提取错误: {e}")
            quality_features = self._get_default_quality_features()
        
        return quality_features
    
    def _get_default_quality_features(self):
        """返回默认的语音质量特征"""
        return {
            'hnr': 0, 'harmonic_ratio': 0,
            'zcr_mean': 0, 'zcr_std': 0, 'zcr_max': 0,
            'jitter': 0, 'shimmer': 0
        }
    
    def extract_temporal_features(self, audio, sr):
        """
        提取时域特征
        
        Args:
            audio: 音频信号
            sr: 采样率
            
        Returns:
            dict: 时域特征
        """
        temporal_features = {}
        
        try:
            intervals = librosa.effects.split(audio, top_db=35)
            
            if len(intervals) > 0:
                voiced_durations = [(end - start) / sr for start, end in intervals]
                total_voiced = sum(voiced_durations)
                total_duration = len(audio) / sr
                
                temporal_features['total_duration'] = float(total_duration)
                temporal_features['voiced_duration'] = float(total_voiced)
                temporal_features['unvoiced_duration'] = float(total_duration - total_voiced)
                temporal_features['voiced_ratio'] = float(total_voiced / total_duration)
                temporal_features['num_voiced_segments'] = len(intervals)
                temporal_features['mean_segment_duration'] = float(np.mean(voiced_durations))
                temporal_features['std_segment_duration'] = float(np.std(voiced_durations))
                
                gaps = []
                for i in range(1, len(intervals)):
                    gap = (intervals[i][0] - intervals[i-1][1]) / sr
                    gaps.append(gap)
                
                if len(gaps) > 0:
                    temporal_features['mean_gap_duration'] = float(np.mean(gaps))
                    temporal_features['std_gap_duration'] = float(np.std(gaps))
                else:
                    temporal_features['mean_gap_duration'] = 0
                    temporal_features['std_gap_duration'] = 0
            else:
                temporal_features = self._get_default_temporal_features(len(audio) / sr)
                
        except Exception as e:
            print(f"时域特征提取错误: {e}")
            temporal_features = self._get_default_temporal_features(len(audio) / sr)
        
        return temporal_features
    
    def _get_default_temporal_features(self, duration):
        """返回默认的时域特征"""
        return {
            'total_duration': float(duration),
            'voiced_duration': 0,
            'unvoiced_duration': float(duration),
            'voiced_ratio': 0,
            'num_voiced_segments': 0,
            'mean_segment_duration': 0,
            'std_segment_duration': 0,
            'mean_gap_duration': 0,
            'std_gap_duration': 0
        }
    
    def extract_segment_features(self, segment, sr):
        """
        提取单个音频段的所有特征（用于并行处理）
        
        Args:
            segment: 音频段
            sr: 采样率
            
        Returns:
            dict: 所有特征
        """
        return self.extract_all_features(segment, sr)