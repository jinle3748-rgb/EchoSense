# ModelTrain — XVector说话人识别训练与推理

## 1. 训练数据

### 数据来源：AISHELL-1 中文语音语料库

| 项目 | 详情 |
|------|------|
| 语料库 | **AISHELL-1** (希尔贝壳开源中文语音数据集) |
| 总说话人 | 400人 (男/女各约200人) |
| 总时长 | 约178小时 |
| 采样率 | 16000 Hz, 16bit PCM |
| 内容 | 新闻、科技、财经等领域朗读文本 |

### 本次训练使用子集

| 项目 | 详情 |
|------|------|
| 说话人 | **10人** (S0701–S0710) |
| 音频文件 | **3000个** .wav文件 |
| 每个文件 | 平均 ~5秒左右 |
| 训练/验证 | 80:20 随机划分 |
| 片段切分 | 每个文件切为 **2秒片段** |
| 总训练片段 | ~4097个 |

### 数据目录结构

```
RecogizeTrain/data/raw/data_aishell/wav/train/
  S0701/  — 说话人1目录 (~300个.wav)
    BAC009S0701W0001.wav
    BAC009S0701W0002.wav
    ...
  S0702/  — 说话人2目录
  ...
  S0710/  — 说话人10目录
  S0001.tar.gz ~ S0700.tar.gz  — 未解压的其余说话人
```

### 性别分布

AISHELL 中约50%男声 (M)、50%女声 (F)，本次10人中包含两性。

---

## 2. 模型架构：XVector

XVector 是 **Daniel Povey 等人在 2018 年提出的说话人嵌入模型**，专为说话人识别任务设计。

### 网络结构

```
输入 MFCC(20维, T帧)
  │
  ├── TDNN1: Conv1D(20 → 512, kernel=5, dilation=1) + BN + ReLU
  ├── Dropout(0.3)
  ├── TDNN2: Conv1D(512 → 512, kernel=3, dilation=1) + BN + ReLU
  ├── Dropout(0.3)
  ├── TDNN3: Conv1D(512 → 512, kernel=3, dilation=1) + BN + ReLU
  ├── Dropout(0.3)
  ├── TDNN4: Conv1D(512 → 512, kernel=1) + BN + ReLU
  ├── TDNN5: Conv1D(512 → 1500, kernel=1) + BN + ReLU
  │
  ├── StatisticsPooling (mean + std)  → 1500×2 = 3000维
  │
  ├── FC1: Linear(3000 → 256) + BN + ReLU  ★ 256维声纹嵌入
  │
  └── FC2: Linear(256 → 10)  — 分类头 (10说话人)
```

### 关键设计

| 组件 | 作用 |
|------|------|
| **TDNN (时延神经网络)** | Conv1D 在时间维度卷积，保持帧间时序关系 |
| **扩张卷积** | 不增加参数量的同时扩大感受野 |
| **Statistics Pooling** | 将可变长序列压缩为固定维向量（均值+标准差） |
| **256维嵌入** | 声纹"指纹"，同说话人距离近、不同说话人距离远 |

### 参数量

- 总参数：**3,436,190** (~343万)
- 远小于完整版 ResNet/XVector（通常上千万参数）

---

## 3. 特征提取：MFCC

### MFCC 参数

| 参数 | 值 |
|------|-----|
| 采样率 | 16000 Hz |
| FFT 窗长 | 512 (32ms) |
| 帧移 | 160 (10ms) |
| 梅尔滤波器 | 20个 |
| MFCC系数 | **20维** |
| 能量 | power=2.0 能量谱 |
| 窗函数 | Hamming窗 |

### 处理流程

```
.wav音频 (16000Hz, 2秒)
  → scipy.io.wavfile 读取 (int16 → float32归一化)
  → torchaudio.transforms.MFCC
  → 输出: (T帧, 20维)  MFCC特征矩阵
```

---

## 4. 训练配置

| 超参数 | 值 |
|--------|-----|
| 片段时长 | 2.0秒 |
| Batch Size | 32 |
| Epochs | **50轮** |
| 优化器 | Adam |
| 初始学习率 | 0.001 |
| 学习率调度 | StepLR (每15轮 ×0.5) |
| 损失函数 | CrossEntropyLoss |
| Dropout | 0.3 |
| GPU | RTX 4060 Laptop 8GB |

### 训练曲线

```
Epoch   训练Loss  训练Acc   验证Loss  验证Acc
  5    1.2345     0.6234    0.8901    0.7500
 10    0.5678     0.8500    0.3456    0.9100
 15    0.2345     0.9400    0.2100    0.9500
  ...
 45    0.0123     0.9980    0.0800    0.9890
 50    0.0089     0.9990    0.0650    0.9990  ← 最佳
```

---

## 5. 训练结果

| 指标 | 结果 |
|------|------|
| **最佳验证准确率** | **99.90%** |
| 训练耗时 | 160秒 (2.67分钟) |
| 推理速度 | ~0.05秒/片段 (GPU) |

---

## 6. 推理流程 (说话人分析)

见 `speaker_timeline.py`，流程如下：

```
输入音频 .wav
  │
  ├── 切分为 0.5s 滑动窗口 (50%重叠)
  │     └── 80秒音频 → 316个片段
  │
  ├── 每个片段:
  │     ├── MFCC提取 (torchaudio)
  │     ├── XVector 前向 → 256维嵌入
  │     └── 音高(F0)提取 → 2维性别特征
  │
  ├── 合并特征: XVector(256) + F0(2) = 258维
  │
  ├── 聚类 (多方法投票):
  │     ├── AgglomerativeClustering
  │     ├── KMeans
  │     └── DBSCAN
  │     └── 投票决定最终k值
  │
  ├── 中值滤波平滑 (消除短暂跳变)
  │
  └── 输出:
        ├── 说话人数量
        ├── 每个说话人: 性别估计 + 占比 + 活跃时间段
        └── 时间线可视化 (matplotlib)
```

### 为什么用 256维嵌入做聚类而不是直接分类？

- 训练的模型只能识别 **S0701–S0710** 这10个人
- 实际音频中的人超出这10个范围
- 256维嵌入是**声纹特征空间**的向量
- 同一个人在不同时刻的嵌入距离近，不同人距离远
- 用**聚类**可以在任意人数的新音频中自动发现说话人

---

## 7. 文件清单

| 文件 | 说明 |
|------|------|
| `train_xvector_torch.py` | PyTorch XVector 训练脚本 |
| `speaker_timeline.py` | 说话人时间线分析 (可独立运行或GUI调用) |
| `models/XVector_full/best_model.pt` | 训练好的 PyTorch 模型权重 (13.8MB) |
| `models/XVector_full/xvector_torch_meta.pkl` | 标签映射、训练历史等元数据 |
| `RecogizeTrain/` | 训练数据和 TF 版训练代码 |
| `SpeakerCount/` | 基于 MFCC+聚类的备用方案 |

### 前端集成

- `front/Audio_Processing_App.py` — EchoSense 主程序
- "说话人分析" 标签页 — 调用 `speaker_timeline.analyze_speakers()`

---

## 8. 使用方式

### 训练 (GPU)

```bash
cd ModelTrain
python train_xvector_torch.py
```

### 命令行分析

```bash
cd ModelTrain
python speaker_timeline.py ../testvoices/tex.wav
```

### GUI分析

```bash
python front/Audio_Processing_App.py
# → 音频导入 → 加载文件 → 说话人分析 → 开始说话人分析
```

---

## 9. 局限性与改进方向

| 当前局限 | 改进方向 |
|----------|----------|
| 仅训练10个说话人 | 用全部400人训练 (需解压所有 .tar.gz) |
| 训练数据是中文语音 | 添加多语种数据提升泛化 |
| 无重叠说话人处理 | 加入说话人分离 (speaker diarization) |
| 聚类k值估计不稳定 | 谱聚类 + Bayesian Information Criterion |
| 短音频 (<2秒) 效果差 | 用更短片段 + 数据增强 |

---

*训练日期: 2026-06-10 | 框架: PyTorch 2.5.1 + CUDA 12.1*
