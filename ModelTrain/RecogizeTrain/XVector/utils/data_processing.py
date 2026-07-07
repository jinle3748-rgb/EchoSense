#!/usr/bin/env python3
"""
数据处理工具

功能：
- 查找音频文件
- 分割数据集
- 提取MFCC特征
- 数据增强
- 批量处理
"""

import os
import numpy as np
import librosa
import random
from sklearn.model_selection import train_test_split


def find_audio_files(directory):
    """
    查找目录中的音频文件
    
    参数:
        directory: 目录路径
    
    返回:
        audio_files: 音频文件列表
    """
    audio_files = []
    
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.wav'):
                audio_files.append(os.path.join(root, file))
    
    return audio_files


def get_speaker_label(file_path):
    """
    从文件路径中提取说话人标签
    
    参数:
        file_path: 文件路径
    
    返回:
        speaker_id: 说话人ID
    """
    # AISHELL数据集的文件路径格式: .../wav/S0002/001.wav
    parts = file_path.split(os.sep)
    
    # 优先寻找以'S'开头且后面跟着4个数字的部分（如 S0025）
    import re
    for part in parts:
        if re.match(r'S\d{4}', part):
            return part
    
    # 尝试从文件名中提取说话人ID (例如 BAC009S0724W0121.wav -> S0724)
    filename = os.path.basename(file_path)
    match = re.search(r'S\d{4}', filename)
    if match:
        return match.group()
    
    # 如果都不行，返回文件名作为标签
    return os.path.basename(file_path).split('.')[0]


def split_data(audio_files, train_ratio=0.7, val_ratio=0.15):
    """
    分割数据集
    
    参数:
        audio_files: 音频文件列表
        train_ratio: 训练集比例
        val_ratio: 验证集比例
    
    返回:
        train_files: 训练集文件
        val_files: 验证集文件
        test_files: 测试集文件
    """
    # 首先分割训练集和剩余部分
    train_files, remaining_files = train_test_split(
        audio_files, 
        train_size=train_ratio, 
        random_state=42,
        shuffle=True
    )
    
    # 然后分割验证集和测试集
    test_ratio = 1.0 - train_ratio - val_ratio
    val_files, test_files = train_test_split(
        remaining_files, 
        train_size=val_ratio/(val_ratio + test_ratio), 
        random_state=42,
        shuffle=True
    )
    
    return train_files, val_files, test_files


def extract_mfcc(audio_path, sr=16000, n_mfcc=40, max_length=300):
    """
    提取MFCC特征
    
    参数:
        audio_path: 音频文件路径
        sr: 采样率
        n_mfcc: MFCC特征维度
        max_length: 最大时间步长
    
    返回:
        mfcc: MFCC特征
    """
    try:
        # 加载音频
        audio, _ = librosa.load(audio_path, sr=sr)
        
        # 提取MFCC特征
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
        
        # 转置为 (时间步长, 特征维度)
        mfcc = mfcc.T
        
        # 标准化
        mean = np.mean(mfcc, axis=0)
        std = np.std(mfcc, axis=0)
        std = np.maximum(std, 1e-8)  # 避免除零
        mfcc = (mfcc - mean) / std
        
        # 填充或截断到固定长度
        if len(mfcc) > max_length:
            mfcc = mfcc[:max_length]
        else:
            pad_width = max_length - len(mfcc)
            mfcc = np.pad(mfcc, ((0, pad_width), (0, 0)), mode='constant')
        
        return mfcc
    except Exception as e:
        print(f"提取特征失败: {audio_path}, 错误: {e}")
        return None


def augment_audio(audio, sr=16000):
    """
    音频数据增强
    
    参数:
        audio: 音频数据
        sr: 采样率
    
    返回:
        augmented_audio: 增强后的音频
    """
    augmented_audio = audio.copy()
    
    # 应用多种增强方法
    # 1. 添加随机噪声
    noise = np.random.randn(len(augmented_audio)) * 0.01
    augmented_audio += noise
    
    # 2. 改变音调
    pitch_shift = random.randint(-3, 3)
    augmented_audio = librosa.effects.pitch_shift(augmented_audio, sr=sr, n_steps=pitch_shift)
    
    # 3. 时间拉伸
    rate = random.uniform(0.8, 1.2)
    augmented_audio = librosa.effects.time_stretch(augmented_audio, rate=rate)
    
    # 4. 音量变化
    volume_factor = random.uniform(0.7, 1.3)
    augmented_audio *= volume_factor
    
    # 5. 随机裁剪
    if len(augmented_audio) > sr:  # 至少1秒
        start = random.randint(0, len(augmented_audio) - sr)
        augmented_audio = augmented_audio[start:start + sr]
    
    return augmented_audio


def extract_mfcc_with_augmentation(audio_path, sr=16000, n_mfcc=40, max_length=300, augment=False):
    """
    提取MFCC特征，支持数据增强
    
    参数:
        audio_path: 音频文件路径
        sr: 采样率
        n_mfcc: MFCC特征维度
        max_length: 最大时间步长
        augment: 是否进行数据增强
    
    返回:
        mfcc: MFCC特征
    """
    try:
        # 加载音频
        audio, _ = librosa.load(audio_path, sr=sr)
        
        # 数据增强
        if augment:
            audio = augment_audio(audio, sr=sr)
        
        # 提取MFCC特征
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
        
        # 转置为 (时间步长, 特征维度)
        mfcc = mfcc.T
        
        # 标准化
        mean = np.mean(mfcc, axis=0)
        std = np.std(mfcc, axis=0)
        std = np.maximum(std, 1e-8)  # 避免除零
        mfcc = (mfcc - mean) / std
        
        # 填充或截断到固定长度
        if len(mfcc) > max_length:
            mfcc = mfcc[:max_length]
        else:
            pad_width = max_length - len(mfcc)
            mfcc = np.pad(mfcc, ((0, pad_width), (0, 0)), mode='constant')
        
        return mfcc
    except Exception as e:
        print(f"提取特征失败: {audio_path}, 错误: {e}")
        return None


def batch_extract_features(audio_files, batch_size=32, n_mfcc=40, max_length=300, augment=False):
    """
    批量提取特征
    
    参数:
        audio_files: 音频文件列表
        batch_size: 批处理大小
        n_mfcc: MFCC特征维度
        max_length: 最大时间步长
        augment: 是否进行数据增强
    
    返回:
        features: 特征数组
        labels: 标签数组
        speaker_to_id: 说话人到ID的映射
    """
    features = []
    labels = []
    speaker_to_id = {}
    current_id = 0
    
    for i, audio_file in enumerate(audio_files):
        # 提取说话人标签
        speaker = get_speaker_label(audio_file)
        if speaker is None:
            continue
        
        # 分配说话人ID
        if speaker not in speaker_to_id:
            speaker_to_id[speaker] = current_id
            current_id += 1
        
        # 提取特征
        mfcc = extract_mfcc_with_augmentation(
            audio_file, 
            n_mfcc=n_mfcc, 
            max_length=max_length, 
            augment=augment
        )
        
        if mfcc is not None:
            features.append(mfcc)
            labels.append(speaker_to_id[speaker])
        
        # 打印进度
        if (i + 1) % 10 == 0:
            print(f"处理进度: {i + 1}/{len(audio_files)}")
        
        # 每处理50个文件，打印一次说话人数量
        if (i + 1) % 50 == 0:
            print(f"当前识别到的说话人数量: {len(speaker_to_id)}")
    
    # 转换为numpy数组
    features = np.array(features)
    labels = np.array(labels)
    
    return features, labels, speaker_to_id
