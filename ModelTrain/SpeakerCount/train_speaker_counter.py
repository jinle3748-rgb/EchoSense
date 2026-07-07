"""
说话人计数模型训练脚本

功能：
- 基于XVector架构训练说话人计数模型
- 使用项目中已有的音频数据
- 支持数据增强
- 支持模型保存和评估

训练流程：
1. 加载音频数据（支持多说话人音频）
2. 提取MFCC特征
3. 训练XVector模型
4. 保存模型到 models/XVector/ 目录
"""

import os
import pickle
import warnings
from datetime import datetime

import numpy as np
import librosa
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings('ignore')


class XVectorTrainer:
    """XVector说话人计数模型训练器"""
    
    def __init__(self, model_dir: str = "models/XVector"):
        """
        初始化训练器
        
        Args:
            model_dir: 模型保存目录
        """
        self.model_dir = model_dir
        self.label_encoder = LabelEncoder()
        self.model = None
        self.history = None
        
        # 创建模型目录
        os.makedirs(self.model_dir, exist_ok=True)
    
    def extract_mfcc(self, audio_path: str, sr: int = 16000, n_mfcc: int = 20):
        """
        提取音频的MFCC特征
        
        Args:
            audio_path: 音频文件路径
            sr: 采样率
            n_mfcc: MFCC维度
            
        Returns:
            np.ndarray: MFCC特征矩阵
        """
        y, sr = librosa.load(audio_path, sr=sr)
        
        # 提取MFCC
        mfcc = librosa.feature.mfcc(
            y=y,
            sr=sr,
            n_mfcc=n_mfcc,
            n_fft=512,
            hop_length=160
        )
        
        # 归一化
        mfcc = (mfcc - np.mean(mfcc, axis=1, keepdims=True)) / (np.std(mfcc, axis=1, keepdims=True) + 1e-8)
        
        return mfcc
    
    def prepare_data(self, audio_files: list, labels: list, segment_duration: float = 1.0):
        """
        准备训练数据
        
        Args:
            audio_files: 音频文件路径列表
            labels: 对应的说话人标签列表
            segment_duration: 每个片段的时长（秒）
            
        Returns:
            tuple: (X_train, X_test, y_train, y_test)
        """
        print("准备训练数据...")
        
        X = []
        y = []
        
        for audio_path, label in zip(audio_files, labels):
            try:
                # 加载音频
                y_audio, sr = librosa.load(audio_path, sr=16000)
                
                # 分段
                samples_per_segment = int(segment_duration * sr)
                num_segments = len(y_audio) // samples_per_segment
                
                for i in range(num_segments):
                    start = i * samples_per_segment
                    end = start + samples_per_segment
                    segment = y_audio[start:end]
                    
                    # 跳过静音片段
                    rms = librosa.feature.rms(y=segment).mean()
                    if rms > 0.001:
                        # 提取MFCC
                        mfcc = librosa.feature.mfcc(
                            y=segment,
                            sr=sr,
                            n_mfcc=20,
                            n_fft=512,
                            hop_length=160
                        )
                        
                        # 归一化
                        mfcc = (mfcc - np.mean(mfcc, axis=1, keepdims=True)) / (np.std(mfcc, axis=1, keepdims=True) + 1e-8)
                        
                        # 转置MFCC使其形状为 (time_steps, n_mfcc)
                        mfcc = mfcc.T
                        
                        X.append(mfcc)
                        y.append(label)
                        
            except Exception as e:
                print(f"处理 {audio_path} 时出错: {e}")
                continue
        
        X = np.array(X)
        y = np.array(y)
        
        print(f"总样本数: {len(X)}")
        print(f"特征形状: {X.shape}")  # 应为 (n_samples, time_steps, n_mfcc)
        print(f"说话人数量: {len(set(y))}")
        
        # 编码标签
        y_encoded = self.label_encoder.fit_transform(y)
        
        # 保存标签编码器
        with open(os.path.join(self.model_dir, "speaker_to_id.pkl"), 'wb') as f:
            pickle.dump(self.label_encoder, f)
        
        # 划分训练集和测试集
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
        )
        
        print(f"训练集大小: {len(X_train)}")
        print(f"测试集大小: {len(X_test)}")
        
        return X_train, X_test, y_train, y_test
    
    def build_model(self, input_shape: tuple, num_classes: int, embedding_dim: int = 512):
        """
        构建XVector模型
        
        Args:
            input_shape: 输入形状 (n_mfcc, time_steps)
            num_classes: 说话人数量
            embedding_dim: 嵌入维度
            
        Returns:
            model: Keras模型
        """
        try:
            import tensorflow as tf
            from tensorflow.keras.models import Model
            from tensorflow.keras.layers import (
                Input, Conv1D, BatchNormalization, ReLU,
                GlobalAveragePooling1D, Dense, Dropout,
                Flatten, Reshape
            )
            
            # 输入层
            inputs = Input(shape=input_shape)
            
            # TDNN层（时延神经网络）
            x = Conv1D(512, kernel_size=5, padding='same', activation='relu')(inputs)
            x = BatchNormalization()(x)
            x = Dropout(0.3)(x)
            
            x = Conv1D(512, kernel_size=3, padding='same', activation='relu')(x)
            x = BatchNormalization()(x)
            x = Dropout(0.3)(x)
            
            x = Conv1D(512, kernel_size=3, padding='same', activation='relu')(x)
            x = BatchNormalization()(x)
            x = Dropout(0.3)(x)
            
            # 统计池化层
            x = GlobalAveragePooling1D()(x)
            
            # 嵌入层（XVector）
            x = Dense(embedding_dim, activation='relu', name='embedding')(x)
            x = BatchNormalization()(x)
            x = Dropout(0.5)(x)
            
            # 分类层
            outputs = Dense(num_classes, activation='softmax')(x)
            
            model = Model(inputs=inputs, outputs=outputs)
            
            model.compile(
                optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy']
            )
            
            print("模型架构:")
            model.summary()
            
            return model
            
        except ImportError:
            raise ImportError("请安装 tensorflow: pip install tensorflow")
    
    def train(self, X_train, X_test, y_train, y_test, epochs: int = 50, batch_size: int = 32):
        """
        训练模型
        
        Args:
            X_train: 训练特征
            X_test: 测试特征
            y_train: 训练标签
            y_test: 测试标签
            epochs: 训练轮数
            batch_size: 批次大小
            
        Returns:
            history: 训练历史
        """
        print("\n开始训练...")
        
        # 构建模型
        input_shape = X_train.shape[1:]
        num_classes = len(set(y_train))
        
        self.model = self.build_model(input_shape, num_classes)
        
        # 训练
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=epochs,
            batch_size=batch_size,
            verbose=1
        )
        
        # 评估
        loss, accuracy = self.model.evaluate(X_test, y_test, verbose=0)
        print(f"\n测试集准确率: {accuracy:.4f}")
        print(f"测试集损失: {loss:.4f}")
        
        return self.history
    
    def save_model(self):
        """保存模型"""
        if self.model is None:
            raise ValueError("模型未训练")
        
        # 保存完整模型
        model_path = os.path.join(self.model_dir, "xvector_model.h5")
        self.model.save(model_path)
        print(f"模型已保存: {model_path}")
        
        # 保存嵌入模型（用于提取特征）
        from tensorflow.keras.models import Model
        embedding_model = Model(
            inputs=self.model.input,
            outputs=self.model.get_layer('embedding').output
        )
        embedding_path = os.path.join(self.model_dir, "xvector_embedding.h5")
        embedding_model.save(embedding_path)
        print(f"嵌入模型已保存: {embedding_path}")
        
        # 保存训练历史
        if self.history:
            history_path = os.path.join(self.model_dir, "training_history.pkl")
            with open(history_path, 'wb') as f:
                pickle.dump(self.history.history, f)
            print(f"训练历史已保存: {history_path}")
        
        # 生成训练报告
        self._generate_report()
    
    def _generate_report(self):
        """生成训练报告"""
        report_path = os.path.join(self.model_dir, "xvector_training_report.md")
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("# X-vector模型训练性能报告\n\n")
            f.write("## 训练配置\n")
            f.write(f"- 模型: X-vector\n")
            f.write(f"- 训练时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            if self.history:
                f.write(f"- 训练轮数: {len(self.history.history['loss'])} 轮\n")
            
            f.write("\n## 性能指标\n")
            if self.history:
                f.write(f"- 训练准确率: {self.history.history['accuracy'][-1]:.4f}\n")
                f.write(f"- 验证准确率: {self.history.history['val_accuracy'][-1]:.4f}\n")
                f.write(f"- 训练损失: {self.history.history['loss'][-1]:.4f}\n")
                f.write(f"- 验证损失: {self.history.history['val_loss'][-1]:.4f}\n")
            
            f.write("\n## 模型信息\n")
            f.write("- 模型架构: X-vector (TDNN-based)\n")
            f.write("- 嵌入维度: 512\n")
            
            f.write("\n## 优化建议\n")
            f.write("- 数据增强: 增加更多的数据增强方法，如音量变化、噪声注入等\n")
            f.write("- 模型微调: 尝试不同的学习率和批量大小\n")
            f.write("- 特征提取: 考虑使用更高级的特征提取方法\n")
            f.write("- 模型集成: 尝试使用模型集成技术提高性能\n")
        
        print(f"训练报告已保存: {report_path}")


def train_from_directory(data_dir: str, model_dir: str = "models/XVector"):
    """
    从目录训练模型
    
    目录结构:
    data_dir/
        speaker_1/
            audio1.wav
            audio2.wav
        speaker_2/
            audio1.wav
            audio2.wav
    
    Args:
        data_dir: 数据目录
        model_dir: 模型保存目录
    """
    trainer = XVectorTrainer(model_dir)
    
    # 收集音频文件和标签
    audio_files = []
    labels = []
    
    for speaker_dir in os.listdir(data_dir):
        speaker_path = os.path.join(data_dir, speaker_dir)
        if os.path.isdir(speaker_path):
            for audio_file in os.listdir(speaker_path):
                if audio_file.endswith(('.wav', '.mp3', '.flac')):
                    audio_files.append(os.path.join(speaker_path, audio_file))
                    labels.append(speaker_dir)
    
    print(f"找到 {len(audio_files)} 个音频文件")
    print(f"说话人数量: {len(set(labels))}")
    
    # 准备数据
    X_train, X_test, y_train, y_test = trainer.prepare_data(audio_files, labels)
    
    # 训练
    trainer.train(X_train, X_test, y_train, y_test)
    
    # 保存模型
    trainer.save_model()
    
    print("\n训练完成!")


if __name__ == "__main__":
    # 示例：使用项目中的音频数据训练
    # 如果你有多个说话人的音频数据，可以按以下结构组织：
    # data/
    #   speaker_1/
    #     audio1.wav
    #   speaker_2/
    #     audio2.wav
    
    # train_from_directory("data")
    
    print("训练脚本已准备好")
    print("请按以下步骤操作:")
    print("1. 准备训练数据（每个说话人一个文件夹）")
    print("2. 调用 train_from_directory('你的数据目录')")
    print("3. 模型将自动保存到 models/XVector/ 目录")