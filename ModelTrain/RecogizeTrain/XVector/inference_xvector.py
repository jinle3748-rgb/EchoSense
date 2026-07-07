#!/usr/bin/env python3
"""
X-vector模型推理脚本

功能：
- 加载训练好的X-vector模型
- 提取音频特征
- 进行说话人识别
- 支持新说话人检测
"""

import os
import sys
import numpy as np
import librosa
import tensorflow as tf
import pickle

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.data_processing import extract_mfcc


class XVectorSpeakerRecognizer:
    """
    X-vector说话人识别器
    """
    
    def __init__(self, model_path='RecogizeTrain/XVector/models/xvector_model.h5',
                 embedding_path='RecogizeTrain/XVector/models/xvector_embedding.h5',
                 speaker_map_path='RecogizeTrain/XVector/models/speaker_to_id.pkl'):
        """
        初始化X-vector说话人识别器
        
        参数:
            model_path: 模型路径
            embedding_path: 嵌入模型路径
            speaker_map_path: 说话人映射路径
        """
        # 加载模型
        self.model = tf.keras.models.load_model(model_path)
        self.embedding_model = tf.keras.models.load_model(embedding_path)
        
        # 加载说话人映射
        with open(speaker_map_path, 'rb') as f:
            self.speaker_to_id = pickle.load(f)
        
        # 创建ID到说话人的映射
        self.id_to_speaker = {v: k for k, v in self.speaker_to_id.items()}
        
        # 保存已知说话人的嵌入
        self.known_embeddings = []
        self.known_speakers = []
        
        print(f"加载模型成功，已知说话人数量: {len(self.speaker_to_id)}")
    
    def extract_embedding(self, audio_path):
        """
        提取音频的X-vector嵌入
        
        参数:
            audio_path: 音频文件路径
        
        返回:
            embedding: X-vector嵌入
        """
        # 提取MFCC特征
        mfcc = extract_mfcc(audio_path, n_mfcc=40, max_length=300)
        if mfcc is None:
            return None
        
        # 扩展维度以匹配模型输入
        mfcc = np.expand_dims(mfcc, axis=0)
        
        # 提取嵌入
        embedding = self.embedding_model.predict(mfcc, verbose=0)
        
        return embedding[0]
    
    def calculate_similarity(self, embedding1, embedding2):
        """
        计算两个嵌入之间的相似度
        
        参数:
            embedding1: 第一个嵌入
            embedding2: 第二个嵌入
        
        返回:
            similarity: 相似度分数
        """
        # 使用余弦相似度
        similarity = np.dot(embedding1, embedding2) / (
            np.linalg.norm(embedding1) * np.linalg.norm(embedding2)
        )
        return similarity
    
    def recognize_speaker(self, audio_path, threshold=0.7):
        """
        识别说话人
        
        参数:
            audio_path: 音频文件路径
            threshold: 相似度阈值
        
        返回:
            result: 识别结果
        """
        # 提取嵌入
        embedding = self.extract_embedding(audio_path)
        if embedding is None:
            return {
                'success': False,
                'message': '无法提取特征'
            }
        
        # 如果没有已知嵌入，先添加当前嵌入
        if len(self.known_embeddings) == 0:
            self.known_embeddings.append(embedding)
            self.known_speakers.append('Unknown_1')
            return {
                'success': True,
                'speaker': 'Unknown_1',
                'is_new': True,
                'similarity': 0.0
            }
        
        # 计算与已知嵌入的相似度
        similarities = []
        for known_embedding in self.known_embeddings:
            similarity = self.calculate_similarity(embedding, known_embedding)
            similarities.append(similarity)
        
        # 找到最相似的说话人
        max_similarity = max(similarities)
        max_index = similarities.index(max_similarity)
        
        if max_similarity >= threshold:
            # 识别为已知说话人
            speaker = self.known_speakers[max_index]
            return {
                'success': True,
                'speaker': speaker,
                'is_new': False,
                'similarity': max_similarity
            }
        else:
            # 识别为新说话人
            new_speaker_id = len(self.known_speakers) + 1
            new_speaker = f'Unknown_{new_speaker_id}'
            self.known_embeddings.append(embedding)
            self.known_speakers.append(new_speaker)
            return {
                'success': True,
                'speaker': new_speaker,
                'is_new': True,
                'similarity': max_similarity
            }
    
    def batch_recognize(self, audio_files, threshold=0.7):
        """
        批量识别说话人
        
        参数:
            audio_files: 音频文件列表
            threshold: 相似度阈值
        
        返回:
            results: 识别结果列表
        """
        results = []
        
        for audio_file in audio_files:
            result = self.recognize_speaker(audio_file, threshold=threshold)
            result['audio_file'] = audio_file
            results.append(result)
        
        return results


def main():
    print("=" * 60)
    print("X-vector说话人识别")
    print("=" * 60)
    
    # 创建识别器
    recognizer = XVectorSpeakerRecognizer()
    
    # 测试音频文件
    test_files = [
        # 这里可以添加测试音频文件路径
    ]
    
    if len(test_files) > 0:
        print("\n批量识别测试:")
        results = recognizer.batch_recognize(test_files)
        
        for result in results:
            if result['success']:
                status = "新说话人" if result['is_new'] else "已知说话人"
                print(f"音频: {os.path.basename(result['audio_file'])}")
                print(f"说话人: {result['speaker']}")
                print(f"状态: {status}")
                print(f"相似度: {result['similarity']:.4f}")
                print("-" * 40)
            else:
                print(f"音频: {os.path.basename(result['audio_file'])}")
                print(f"错误: {result['message']}")
                print("-" * 40)
    else:
        print("\n请添加测试音频文件路径到 test_files 列表")
    
    print("\n" + "=" * 60)
    print("识别完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()
