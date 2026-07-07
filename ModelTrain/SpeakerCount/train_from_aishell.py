"""
使用AIShell数据集训练说话人计数模型

AIShell数据集结构:
    data_aishell/wav/train/
        S0666/
            BAC009S0666W0433.wav
            BAC009S0666W0434.wav
            ...
        S0701/
            BAC009S0701W0121.wav
            ...
"""

import os
import sys
import tarfile
import glob

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from SpeakerCount.train_speaker_counter import XVectorTrainer


def extract_tar_files(data_dir: str):
    """解压所有tar.gz文件"""
    train_dir = os.path.join(data_dir, "wav", "train")
    
    if not os.path.exists(train_dir):
        print(f"目录不存在: {train_dir}")
        return
    
    tar_files = glob.glob(os.path.join(train_dir, "*.tar.gz"))
    
    if not tar_files:
        print("没有找到tar.gz文件，可能已经解压过了")
        return
    
    print(f"找到 {len(tar_files)} 个压缩包，开始解压...")
    
    for tar_file in tar_files:
        speaker_id = os.path.basename(tar_file).replace('.tar.gz', '')
        extract_dir = os.path.join(train_dir, speaker_id)
        
        if os.path.exists(extract_dir):
            print(f"  {speaker_id} 已解压，跳过")
            continue
        
        try:
            with tarfile.open(tar_file, 'r:gz') as tar:
                tar.extractall(train_dir)
            print(f"  ✓ {speaker_id} 解压完成")
        except Exception as e:
            print(f"  ✗ {speaker_id} 解压失败: {e}")
    
    print("解压完成!")


def collect_aishell_data(data_dir: str, max_speakers: int = None, max_files_per_speaker: int = None):
    """
    收集AIShell数据集
    
    Args:
        data_dir: 数据目录 (data_aishell)
        max_speakers: 最大说话人数量（用于快速测试）
        max_files_per_speaker: 每个说话人最大文件数
        
    Returns:
        tuple: (audio_files, labels)
    """
    train_dir = os.path.join(data_dir, "wav", "train")
    
    audio_files = []
    labels = []
    
    # 获取所有说话人目录
    speaker_dirs = [d for d in os.listdir(train_dir) 
                    if os.path.isdir(os.path.join(train_dir, d))]
    
    if max_speakers:
        speaker_dirs = speaker_dirs[:max_speakers]
    
    print(f"找到 {len(speaker_dirs)} 个说话人")
    
    for speaker_id in speaker_dirs:
        speaker_dir = os.path.join(train_dir, speaker_id)
        wav_files = glob.glob(os.path.join(speaker_dir, "*.wav"))
        
        if max_files_per_speaker:
            wav_files = wav_files[:max_files_per_speaker]
        
        for wav_file in wav_files:
            audio_files.append(wav_file)
            labels.append(speaker_id)
    
    print(f"总音频文件数: {len(audio_files)}")
    print(f"说话人数量: {len(set(labels))}")
    
    return audio_files, labels


def train_from_aishell(data_dir: str = "RecogizeTrain/data/raw/data_aishell",
                       model_dir: str = "models/XVector",
                       max_speakers: int = None,
                       max_files_per_speaker: int = None,
                       epochs: int = 50,
                       batch_size: int = 32):
    """
    使用AIShell数据集训练模型
    
    Args:
        data_dir: AIShell数据目录
        model_dir: 模型保存目录
        max_speakers: 最大说话人数量（None表示全部）
        max_files_per_speaker: 每个说话人最大文件数
        epochs: 训练轮数
        batch_size: 批次大小
    """
    print("="*60)
    print("使用AIShell数据集训练说话人计数模型")
    print("="*60)
    
    # 1. 解压数据
    print("\n1. 检查并解压数据...")
    extract_tar_files(data_dir)
    
    # 2. 收集数据
    print("\n2. 收集训练数据...")
    audio_files, labels = collect_aishell_data(
        data_dir,
        max_speakers=max_speakers,
        max_files_per_speaker=max_files_per_speaker
    )
    
    if len(audio_files) == 0:
        print("没有找到训练数据!")
        return
    
    # 3. 准备数据
    print("\n3. 准备数据...")
    trainer = XVectorTrainer(model_dir)
    X_train, X_test, y_train, y_test = trainer.prepare_data(audio_files, labels)
    
    # 4. 训练
    print("\n4. 开始训练...")
    trainer.train(X_train, X_test, y_train, y_test, epochs=epochs, batch_size=batch_size)
    
    # 5. 保存
    print("\n5. 保存模型...")
    trainer.save_model()
    
    print("\n" + "="*60)
    print("训练完成!")
    print("="*60)


if __name__ == "__main__":
    # 快速测试（使用10个说话人，每人20个文件）
    # train_from_aishell(max_speakers=10, max_files_per_speaker=20, epochs=10)
    
    # 完整训练（使用所有数据）
    # train_from_aishell(epochs=50)
    
    print("训练脚本已准备好")
    print("\n使用方式:")
    print("1. 快速测试（数据少，训练快）:")
    print("   train_from_aishell(max_speakers=10, max_files_per_speaker=20, epochs=10)")
    print("\n2. 完整训练（数据多，效果好）:")
    print("   train_from_aishell(epochs=50)")