#!/usr/bin/env python3
"""
X-vector模型数据处理和分析脚本

功能：
- 查找音频文件
- 提取特征
- 分析数据分布
- 生成数据加载报告
"""

print("脚本开始执行...")

import os
print("导入os完成")
import sys
print("导入sys完成")
import numpy as np
print("导入numpy完成")
import time
print("导入time完成")

print("导入完成...")

# 查找音频文件
def find_audio_files(directory):
    """查找目录中的音频文件"""
    audio_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.wav'):
                audio_files.append(os.path.join(root, file))
    # 随机打乱文件列表，确保从多个文件夹中选择文件
    import random
    random.shuffle(audio_files)
    return audio_files

# 从文件路径中提取说话人标签
def get_speaker_label(file_path):
    """从文件路径中提取说话人标签"""
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

# 简单的特征提取函数（模拟）
def extract_simple_feature(audio_file):
    """提取简单的特征"""
    # 模拟特征提取，返回随机特征
    return np.random.rand(300, 40)

# 从指定目录加载数据
def load_data_from_directory(directory, max_files=None):
    """
    从指定目录加载数据
    
    参数:
        directory: 目录路径
        max_files: 最大文件数量
    
    返回:
        features: 特征数组
        labels: 标签数组
        speaker_to_id: 说话人到ID的映射
    """
    audio_files = find_audio_files(directory)
    print(f"在 {directory} 中找到 {len(audio_files)} 个音频文件")
    
    if max_files and len(audio_files) > max_files:
        audio_files = audio_files[:max_files]
        print(f"限制处理文件数量为: {max_files}")
    
    # 打印前5个文件路径，用于调试
    print(f"前5个文件路径:")
    for i, audio_file in enumerate(audio_files[:5]):
        print(f"  {i+1}: {audio_file}")
        speaker = get_speaker_label(audio_file)
        print(f"  提取的说话人: {speaker}")
    
    # 提取特征和标签
    features = []
    labels = []
    speakers = []
    
    for i, audio_file in enumerate(audio_files):
        # 提取说话人标签
        speaker = get_speaker_label(audio_file)
        if speaker is None:
            continue
        
        # 提取特征
        feature = extract_simple_feature(audio_file)
        
        features.append(feature)
        speakers.append(speaker)
        
        # 打印进度
        if (i + 1) % 100 == 0:
            print(f"处理进度: {i + 1}/{len(audio_files)}")
    
    # 打印说话人列表
    print(f"\n识别到的说话人: {list(set(speakers))}")
    print(f"说话人数量: {len(set(speakers))}")
    
    # 创建说话人到ID的映射
    speaker_to_id = {}
    current_id = 0
    for speaker in speakers:
        if speaker not in speaker_to_id:
            speaker_to_id[speaker] = current_id
            current_id += 1
    
    # 转换标签
    labels = [speaker_to_id[speaker] for speaker in speakers]
    
    # 转换为numpy数组
    features = np.array(features)
    labels = np.array(labels)
    
    return features, labels, speaker_to_id

# 生成数据加载报告
def generate_data_report(train_features, val_features, test_features, speaker_to_id, extract_time, train_dir, dev_dir, test_dir):
    """
    生成数据加载报告
    
    参数:
        train_features: 训练集特征
        val_features: 验证集特征
        test_features: 测试集特征
        speaker_to_id: 说话人到ID的映射
        extract_time: 数据加载时间
        train_dir: 训练目录
        dev_dir: 验证目录
        test_dir: 测试目录
    """
    report = f"""# X-vector模型数据加载报告

## 数据加载结果
- 训练集: {len(train_features)} 个样本
- 验证集: {len(val_features)} 个样本
- 测试集: {len(test_features)} 个样本
- 说话人数量: {len(speaker_to_id)}
- 数据加载时间: {extract_time:.2f} 秒

## 数据目录
- 训练目录: {train_dir}
- 验证目录: {dev_dir}
- 测试目录: {test_dir}

## 特征形状
- 训练集特征形状: {train_features.shape if len(train_features) > 0 else 'N/A'}
- 验证集特征形状: {val_features.shape if len(val_features) > 0 else 'N/A'}
- 测试集特征形状: {test_features.shape if len(test_features) > 0 else 'N/A'}

## 说话人列表（前20个）
{list(speaker_to_id.keys())[:20]}

## 数据分布分析
- 平均每个说话人的样本数: {len(train_features) / len(speaker_to_id):.2f}

## 预处理步骤
1. 从指定目录加载音频文件
2. 随机打乱文件顺序，确保从多个文件夹中选择文件
3. 提取说话人标签
4. 生成模拟特征（实际应用中应使用真实的MFCC特征）
5. 创建说话人到ID的映射
6. 转换标签为数字格式

## 后续建议
1. 使用真实的MFCC特征提取
2. 增加数据增强方法
3. 实现完整的X-vector模型
4. 进行模型训练和评估
5. 生成详细的训练报告
"""
    
    report_path = 'RecogizeTrain/XVector/data_loading_report.md'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"数据加载报告已生成: {report_path}")


def main():
    print("=" * 60)
    print("X-vector模型数据处理")
    print("=" * 60)
    
    # 1. 定义数据集路径
    print("\n[1/4] 加载数据集...")
    data_base_path = "C:\\Users\\27946\\Desktop\\Sound\\RecogizeTrain\\data\\raw\\data_aishell\\wav"
    train_dir = os.path.join(data_base_path, "train")
    dev_dir = os.path.join(data_base_path, "dev")
    test_dir = os.path.join(data_base_path, "test")
    
    # 检查目录是否存在
    for dir_path in [train_dir, dev_dir, test_dir]:
        if not os.path.exists(dir_path):
            print(f"错误：目录不存在: {dir_path}")
            return
    
    print("所有目录存在")
    
    # 2. 加载训练数据
    print("\n[2/4] 加载训练数据...")
    start_time = time.time()
    train_features, train_labels, speaker_to_id = load_data_from_directory(train_dir, max_files=100)
    
    # 3. 加载验证数据
    print("\n[3/4] 加载验证数据...")
    val_features, val_labels, _ = load_data_from_directory(dev_dir, max_files=30)
    
    # 4. 加载测试数据
    print("\n[4/4] 加载测试数据...")
    test_features, test_labels, _ = load_data_from_directory(test_dir, max_files=30)
    
    # 数据加载完成
    extract_time = time.time() - start_time
    print(f"\n数据加载时间: {extract_time:.2f} 秒")
    print(f"训练集: {len(train_features)} 个样本")
    print(f"验证集: {len(val_features)} 个样本")
    print(f"测试集: {len(test_features)} 个样本")
    print(f"说话人数量: {len(speaker_to_id)}")
    
    # 检查数据有效性
    if len(train_features) == 0:
        print("错误：训练数据加载失败，没有有效的特征数据")
        return
    
    if len(speaker_to_id) < 2:
        print(f"错误：说话人数量不足 ({len(speaker_to_id)})，无法进行分类训练")
        return
    
    # 生成数据加载报告
    print("\n[5/5] 生成数据加载报告...")
    generate_data_report(train_features, val_features, test_features, speaker_to_id, extract_time, train_dir, dev_dir, test_dir)
    
    print("\n" + "=" * 60)
    print("X-vector模型数据处理完成！")
    print(f"数据加载报告已生成: RecogizeTrain/XVector/data_loading_report.md")
    print("=" * 60)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback
        print(f"错误: {e}")
        traceback.print_exc()
