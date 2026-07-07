"""
SpeakerCount - 说话人计数模块

功能：
- 识别音频中的说话人数量
- 说话人分离（Speaker Diarization）
- 说话人时间线分析
- 支持GPU加速
- 支持离线部署

两种模式：
1. Pyannote模式：使用pyannote.audio（需要网络下载模型或离线模型）
2. XVector模式：使用项目中已有的XVector模型（完全离线）

使用方式：

# XVector模式（完全离线，推荐）
from SpeakerCount import XVectorSpeakerCounter
counter = XVectorSpeakerCounter()
result = counter.count_speakers("audio.wav")

# Pyannote在线模式（需要网络和HF Token）
from SpeakerCount import SpeakerCounter
counter = SpeakerCounter(hf_token="your_token")
result = counter.count_speakers("audio.wav")

# Pyannote离线模式（使用本地模型）
from SpeakerCount import SpeakerCounter
counter = SpeakerCounter(local_model_path="models/speaker-diarization-3.1")
result = counter.count_speakers("audio.wav")
"""

from .speaker_counter import SpeakerCounter, quick_count_speakers, download_models_for_offline
from .speaker_counter_xvector import XVectorSpeakerCounter

__all__ = [
    'SpeakerCounter',
    'XVectorSpeakerCounter',
    'quick_count_speakers',
    'download_models_for_offline'
]

__version__ = '1.0.0'