# EchoSense 项目文件结构说明

## 一、入口与配置

| 文件 | 作用 |
|------|------|
| `front/Audio_Processing_App.py` | **主程序入口**。PyQt5 桌面 GUI，包含 6 个标签页：音频导入、降噪、性别识别、人声分析、XVector 说话人分析、ECAPA 高级声纹。运行 `py front/Audio_Processing_App.py` 启动。 |
| `requirements.txt` | Python 依赖清单，运行 `pip install -r requirements.txt` 一键安装。 |
| `.gitignore` | Git 忽略规则，排除模型文件、日志、缓存等。 |
| `EchoSense.jpg` | 应用图标，GUI 启动时加载为窗口图标。 |

---

## 二、功能模块

### 1. front/ — 前端界面

| 文件 | 作用 |
|------|------|
| `Audio_Processing_App.py` | EchoSense 主窗口。定义 `MainWindow`、6 个标签页类、工作线程。拖拽/选择音频后串联所有分析模块。 |
| `人声分析功能说明.md` | 人声分析标签页的使用说明文档。 |

### 2. DeNoise/ — 音频降噪

| 文件 | 作用 |
|------|------|
| `__init__.py` | 包初始化，导出 `Denoiser` 类。 |
| `denoiser.py` | 降噪核心。统一接口支持三种算法：传统谱减法、`noisereduce` 高级降噪（推荐）、深度学习降噪。GUI 降噪标签页直接调用。 |

### 3. Gender/ — 性别识别

| 文件 | 作用 |
|------|------|
| `__init__.py` | 包初始化，导出三个分析器。 |
| `GPU_GenderRecognition.py` | **主模块**。`GPUGenderAnalyzer` 类，GPU 加速性别识别（基于基频、共振峰等多维特征）。GUI 性别标签页直接调用。 |
| `GenderRecognition.py` | 备用方案。`AdvancedGenderAnalyzer` 类，基于更丰富的多维声学特征做性别分类，输出详细分析报告。 |
| `ParallelGenderRecognition.py` | 批量处理。`ParallelGenderAnalyzer` 类，封装 `GPUGenderAnalyzer` + 多进程，支持一次分析多个音频文件。 |

### 4. VoiceAnalysis/ — 人声分析（核心）

| 文件 | 作用 |
|------|------|
| `__init__.py` | 包初始化，导出所有核心类。 |
| `voice_analyzer.py` | **主分析器**。`VoiceAnalyzer` 类，整合声学特征提取 + 情绪分析 + 可视化报告，提供 `analyze_audio()` 统一入口。GUI 人声分析标签页直接调用。 |
| `acoustic_features.py` | 特征提取引擎。`AcousticFeatureExtractor` 类，提取基频(F0)、共振峰、频谱质心/带宽/滚降、MFCC(13维+Delta+DeltaDelta)、RMS 能量、过零率、HNR、Jitter、Shimmer 等。 |
| `parallel_processor.py` | 并行加速。`ParallelProcessor` 类，基于 `ProcessPoolExecutor` 多进程并行处理音频段，自动检测 CPU 核心数，也支持 CuPy GPU 加速（可选）。 |
| `logger.py` | 日志系统。提供 `setup_logger`、`log_analysis_start`、`log_error` 等函数，输出到文件和控制台。GUI 也使用此模块记录操作日志。 |

### 5. EmotionAnalysis/ — 情绪分析

| 文件 | 作用 |
|------|------|
| `__init__.py` | 包初始化，导出 `EmotionAnalyzer` 和 `EmotionVisualizer`。 |
| `emotion_analyzer.py` | `EmotionAnalyzer` 类。从声学特征推断 7 种情绪（中性/快乐/悲伤/愤怒/恐惧/惊讶/厌恶），输出情绪分布、强度、置信度和时间线。 |
| `emotion_visualizer.py` | `EmotionVisualizer` 类。生成情绪分析可视化图表：饼图、时间序列折线图、雷达图等。 |

### 6. ECAPA-TDNN/ — 高级声纹识别

| 文件 | 作用 |
|------|------|
| `ecapa_speaker.py` | **说话人分离引擎**。基于 SpeechBrain 预训练的 ECAPA-TDNN 模型（VoxCeleb 数据集），提取 192 维声纹嵌入 → KMeans 聚类 → 轮廓系数自动确定说话人数 → 静音检测 → F0 性别估计 → 后验合并。GUI "高级声纹" 标签页调用。需 `torch` + `speechbrain`。 |

---

## 三、训练模块 (ModelTrain/)

### 3.1 说话人时间线（GUI 依赖）

| 文件 | 作用 |
|------|------|
| `speaker_timeline.py` | **GUI 直接依赖**。`analyze_speakers()` 函数：音频分段 → MFCC 提取 → PyTorch XVector 声纹嵌入 → 多聚类（KMeans+层次聚类+DBSCAN 投票）→ F0 性别估计 → 时间线构建。GUI "说话人分析(XVector)" 标签页调用。 |

### 3.2 XVector 训练脚本

| 文件 | 作用 |
|------|------|
| `train_xvector_torch.py` | **v1** — PyTorch 版 XVector 训练。定义 TDNN 模型 + AishellDataset 数据加载器 + 完整训练循环。最基础的训练入口。 |
| `train_xvector_v2.py` | **v2** — 改进版。先预提取所有音频的 MFCC 特征缓存为 `.npy` 文件，再从缓存训练，加速重复实验。 |
| `train_xvector_v3.py` | **v3** — 最快版。从 v2 生成的 pickle 数据直接训练，跳过特征提取阶段。 |
| `train_final.py` | **最终版**。自包含的 25 人 XVector 训练脚本，从 MFCC 缓存加载，保存为 `.pt` 模型包。由 `speaker_timeline.py` 加载使用。 |

### 3.3 XVector 模型定义与推理

| 文件 | 作用 |
|------|------|
| `RecogizeTrain/XVector/xvector_model.py` | TensorFlow 版 XVector 模型架构定义（TDNN + 统计池化 + 128 维嵌入）。参考实现。 |
| `RecogizeTrain/XVector/inference_xvector.py` | TensorFlow 版推理器。`XVectorSpeakerRecognizer` 类，加载 `.h5` 模型做说话人识别。 |
| `RecogizeTrain/XVector/train_xvector.py` | TensorFlow 版训练脚本（AISHELL 数据集），含数据加载、特征提取、训练报告生成。 |
| `RecogizeTrain/XVector/utils/data_processing.py` | 数据处理工具：音频文件搜索、数据集分割、MFCC 提取、数据增强。 |

### 3.4 XVector 模型文件

| 文件 | 作用 |
|------|------|
| `RecogizeTrain/XVector/models/xvector_model.h5` | 完整 XVector 分类模型（TensorFlow Keras 格式） |
| `RecogizeTrain/XVector/models/xvector_embedding.h5` | XVector 嵌入提取模型（仅编码器部分） |
| `RecogizeTrain/XVector/models/xvector_best.h5` | 训练过程中验证集表现最佳的模型快照 |
| `RecogizeTrain/XVector/models/xvector_model_simple.h5` | 简化版模型 |
| `RecogizeTrain/XVector/models/speaker_to_id.pkl` | 说话人标签到 ID 的映射字典 |

### 3.5 XVector 文档与报告

| 文件 | 作用 |
|------|------|
| `RecogizeTrain/XVector/README.md` | XVector 模块说明文档 |
| `RecogizeTrain/XVector/data_loading_report.md` | 数据加载报告（训练/验证/测试集统计） |
| `RecogizeTrain/XVector/xvector_training_report.md` | 训练性能报告（准确率、损失曲线等） |
| `RecogizeTrain/XVector/xvector_training_report_simple.md` | 简化版训练报告 |
| `RecogizeTrain/XVector/xvector_training_history.png` | 训练历史曲线图（loss/accuracy） |

### 3.6 说话人计数 (SpeakerCount/)

| 文件 | 作用 |
|------|------|
| `__init__.py` | 包初始化，导出 `SpeakerCounter`、`XVectorSpeakerCounter` 等。 |
| `speaker_counter.py` | **在线方案**。`SpeakerCounter` 类，基于 pyannote.audio 的 `speaker-diarization-3.1` 模型，需要 HuggingFace Token。 |
| `speaker_counter_xvector.py` | **离线方案**。`XVectorSpeakerCounter` 类，使用本地 XVector 模型 + DBSCAN/层次聚类，无需网络。 |
| `train_speaker_counter.py` | TensorFlow 版说话人计数模型训练，含数据加载、MFCC 提取、训练、评估、保存。 |
| `train_speaker_counter_gpu.py` | CuPy GPU 加速版训练（纯 GPU 神经网络）。 |
| `train_from_aishell.py` | AISHELL 数据集适配：tar 解压、数据收集、委托训练。 |
| `README.md` | 训练模块说明文档。 |

---

## 四、文档

| 文件 | 作用 |
|------|------|
| `README.md` | 项目主文档：功能介绍、项目结构、快速开始、使用指南、技术架构。 |
| `EMOTION_ANALYSIS_README.md` | 情绪分析模块详细说明：7 种情绪、特征维度、使用示例。 |
| `PROJECT_ORGANIZATION_SUMMARY.md` | 项目重组记录：EmotionAnalysis 独立、目录优化等历史变更。 |
| `PROJECT_STRUCTURE.md` | **本文档** — 每个文件/文件夹的作用说明。 |

---

## 五、数据流关系图

```
Audio_Processing_App.py (GUI 入口)
    │
    ├── 音频导入标签页 → librosa 加载 → 显示文件信息
    │
    ├── 降噪标签页 → DeNoise/denoiser.py
    │       └── 支持：谱减法 / noisereduce / 深度学习
    │
    ├── 性别标签页 → Gender/GPU_GenderRecognition.py
    │       └── GPU 加速多维特征 → 男女评分时间序列
    │
    ├── 人声分析标签页 → VoiceAnalysis/voice_analyzer.py
    │       ├── acoustic_features.py → 声学特征提取
    │       ├── EmotionAnalysis/emotion_analyzer.py → 情绪分析
    │       └── parallel_processor.py → 并行加速
    │
    ├── 说话人分析标签页 → ModelTrain/speaker_timeline.py
    │       └── XVector 嵌入 + 多聚类 → 说话人时间线
    │
    └── 高级声纹标签页 → ECAPA-TDNN/ecapa_speaker.py
            └── ECAPA-TDNN 嵌入 + KMeans → 说话人分离
```

---

## 六、运行方式

```powershell
# 1. 安装依赖（仅首次）
py -m pip install -r requirements.txt

# 2. 启动 GUI
cd front
py Audio_Processing_App.py
```

**可选依赖：**
- `torch` + `speechbrain` → 启用 ECAPA 高级声纹功能
- `cupy` → 启用 GPU 加速
- `tensorflow` → 启用 XVector 模型推理（Python 3.14 暂不支持）
