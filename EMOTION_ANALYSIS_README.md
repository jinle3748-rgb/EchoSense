# 情绪分析模块使用说明

## 概述

已成功将情绪分析相关功能整理到独立的 `EmotionAnalysis` 文件夹中。

## 目录结构

```
Sound/
├── EmotionAnalysis/           # 新增：情绪分析模块文件夹
│   ├── __init__.py            # 模块初始化
│   ├── emotion_analyzer.py    # 情绪分析器
│   └── emotion_visualizer.py  # 情绪可视化器
├── VoiceAnalysis/             # 原有：人声分析模块（保留）
│   ├── __init__.py
│   ├── acoustic_features.py
│   ├── emotion_analyzer.py
│   ├── voice_analyzer.py
│   ├── visualizer.py
│   └── ...
└── ...
```

## 模块功能

### 1. EmotionAnalyzer（情绪分析器）

**文件位置**：`EmotionAnalysis/emotion_analyzer.py`

**功能**：
- 基于声学特征的情绪识别
- 情绪强度分析
- 情绪变化趋势分析
- 多维度情绪评估

**支持的情绪类别**：
- 中性 (neutral)
- 快乐 (happy)
- 悲伤 (sad)
- 愤怒 (angry)
- 恐惧 (fear)
- 惊讶 (surprise)
- 厌恶 (disgust)

**主要方法**：
- `analyze_emotion(features)` - 分析单个情绪
- `analyze_emotion_timeline(features_timeline)` - 分析情绪时间线
- `get_emotion_description(emotion, intensity, confidence)` - 获取情绪描述

### 2. EmotionVisualizer（情绪可视化器）

**文件位置**：`EmotionAnalysis/emotion_visualizer.py`

**功能**：
- 情绪分布可视化（饼图+柱状图）
- 情绪时间线可视化
- 情绪强度图表
- 综合情绪报告

**主要方法**：
- `plot_emotion_distribution(emotion_distribution, audio_file, save_path)` - 绘制情绪分布图
- `plot_emotion_timeline(emotion_timeline, audio_file, save_path)` - 绘制情绪时间线
- `plot_emotion_summary(emotion_result, audio_file, save_path)` - 绘制综合报告

## 使用方法

### 基本用法

```python
from EmotionAnalysis import EmotionAnalyzer, EmotionVisualizer

# 初始化
analyzer = EmotionAnalyzer()
visualizer = EmotionVisualizer(output_dir="./results")

# 分析情绪
result = analyzer.analyze_emotion(features)

# 可视化结果
visualizer.plot_emotion_summary(result, "audio.wav")
```

### 从 GUI 中使用

在 `front/Audio_Processing_App.py` 中已经添加了导入支持，可以直接使用：

```python
from EmotionAnalysis.emotion_analyzer import EmotionAnalyzer
from EmotionAnalysis.emotion_visualizer import EmotionVisualizer
```

## 兼容性说明

### VoiceAnalysis 模块

原有的 `VoiceAnalysis` 模块保持完全不变，您仍然可以正常使用：

```python
from VoiceAnalysis.voice_analyzer import VoiceAnalyzer
```

### 向后兼容

- 所有现有功能保持完整
- 原有的导入语句继续有效
- 新增的模块为独立功能，不影响现有代码

## 测试

运行测试脚本验证模块：

```bash
cd Sound
python test_emotion_module.py
```

## 相关文件

- `test_emotion_module.py` - 情绪分析模块测试脚本
- `front/Audio_Processing_App.py` - GUI应用（已更新导入）
- `VoiceAnalysis/` - 原有人声分析模块（保留完整功能）
- `EMOTION_ANALYSIS_README.md` - 本文档
