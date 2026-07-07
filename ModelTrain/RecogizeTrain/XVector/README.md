# X-vector 声纹识别系统

本系统实现了基于X-vector的声纹识别模型，用于说话人识别和验证。X-vector是一种先进的声纹识别技术，基于TDNN（时间延迟神经网络），能够提取固定长度的说话人嵌入向量。

## 目录结构

```
XVector/
├── xvector_model.py       # X-vector模型定义
├── train_xvector.py       # 训练脚本
├── inference_xvector.py   # 推理脚本
├── README.md              # 本说明文件
├── utils/
│   └── data_processing.py # 数据处理工具
└── models/                # 模型保存目录
```

## 依赖项

- Python 3.7+
- TensorFlow 2.4+
- NumPy
- Librosa
- Matplotlib
- Scikit-learn

## 安装依赖

```bash
pip install tensorflow numpy librosa matplotlib scikit-learn
```

## 数据准备

本系统使用AISHELL-1数据集进行训练。请按照以下步骤准备数据：

1. 下载AISHELL-1数据集：https://www.openslr.org/33/
2. 解压数据集到 `RecogizeTrain/data/raw` 目录
3. 确保目录结构如下：
   ```
   RecogizeTrain/data/raw/
   └── data_aishell/
       └── wav/
           ├── S0002/
           ├── S0003/
           └── ...
   ```

## 模型训练

运行训练脚本开始训练X-vector模型：

```bash
cd RecogizeTrain/XVector
python train_xvector.py
```

训练过程会：
1. 检查系统资源（GPU可用性）
2. 查找音频文件
3. 分割数据集（训练集70%，验证集15%，测试集15%）
4. 提取MFCC特征（支持数据增强）
5. 训练X-vector模型
6. 保存模型和说话人映射
7. 生成性能报告和训练历史图表

## 模型推理

使用训练好的模型进行说话人识别：

```bash
cd RecogizeTrain/XVector
python inference_xvector.py
```

推理系统支持：
1. 加载训练好的X-vector模型
2. 提取音频的X-vector嵌入
3. 计算与已知说话人的相似度
4. 识别说话人（包括新说话人检测）
5. 批量处理多个音频文件

## 技术特点

1. **先进的模型架构**：基于TDNN的X-vector模型，提取固定长度的说话人嵌入
2. **数据增强**：支持噪声注入、音调变化和时间拉伸等数据增强方法
3. **GPU加速**：自动检测GPU并使用GPU加速训练
4. **批量处理**：支持批量特征提取和模型训练，提高效率
5. **新说话人检测**：通过相似度阈值自动检测新说话人
6. **详细的性能报告**：生成训练历史和性能分析图表

## 模型输出

训练完成后，系统会生成以下文件：

- `models/xvector_model.h5` - 完整的X-vector模型
- `models/xvector_embedding.h5` - 用于提取嵌入的模型
- `models/speaker_to_id.pkl` - 说话人到ID的映射
- `models/xvector_model.png` - 模型架构图
- `xvector_training_report.md` - 训练性能报告
- `xvector_training_history.png` - 训练历史图表

## 应用场景

- **说话人识别**：识别音频中的说话人
- **说话人验证**：验证说话人身份
- **说话人分离**：在多说话人场景中分离不同说话人
- **声纹认证**：基于声纹的身份认证

## 注意事项

1. 训练模型需要较大的计算资源，建议使用GPU加速
2. 数据集大小会影响训练时间和模型性能
3. 可以根据实际情况调整模型参数和训练配置
4. 对于新的说话人，系统会自动检测并分配新的ID

## 参考资料

- X-vector: Robust DNN Embeddings for Speaker Recognition
- AISHELL-1: An Open-Source Mandarin Speech Corpus
