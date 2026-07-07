"""
说话人计数模块

功能：
- 识别音频中有多少人在说话
- 说话人分离（Speaker Diarization）
- 说话人时间线分析
- 支持GPU加速
- 支持离线部署

依赖：
- pyannote.audio
- Hugging Face Token（在线模式需要）

离线部署：
1. 在有网络的机器上下载模型（约32MB）
2. 使用 local_model_path 参数指定本地模型路径
"""

import os
import warnings
from typing import Optional, Dict, List, Tuple, Union

warnings.filterwarnings('ignore')


class SpeakerCounter:
    """说话人计数器"""
    
    def __init__(self, 
                 hf_token: Optional[str] = None,
                 use_gpu: bool = True,
                 model_name: str = "pyannote/speaker-diarization-3.1",
                 local_model_path: Optional[str] = None):
        """
        初始化说话人计数器
        
        Args:
            hf_token: Hugging Face访问令牌（在线模式需要）
                      如果未提供，将尝试从环境变量HF_TOKEN读取
            use_gpu: 是否使用GPU加速
            model_name: 使用的模型名称（在线模式）
            local_model_path: 本地模型目录路径（离线模式）
                             例如: "models/speaker-diarization-3.1"
        """
        self.hf_token = hf_token or os.environ.get("HF_TOKEN")
        self.use_gpu = use_gpu
        self.model_name = model_name
        self.local_model_path = local_model_path
        self.pipeline = None
        self.device = None
        
        self._init_pipeline()
    
    def _init_pipeline(self):
        """初始化pyannote pipeline"""
        try:
            import torch
            from pyannote.audio import Pipeline
            
            # 设置设备
            if self.use_gpu and torch.cuda.is_available():
                self.device = torch.device("cuda")
                print(f"使用GPU: {torch.cuda.get_device_name(0)}")
            else:
                self.device = torch.device("cpu")
                print("使用CPU")
            
            # 加载pipeline
            if self.local_model_path:
                # 离线模式：使用本地模型
                if os.path.exists(self.local_model_path):
                    print(f"使用本地模型: {self.local_model_path}")
                    self.pipeline = Pipeline.from_pretrained(self.local_model_path)
                else:
                    raise ValueError(
                        f"本地模型路径不存在: {self.local_model_path}\n"
                        "请先下载模型到本地目录"
                    )
            else:
                # 在线模式：从Hugging Face下载
                if self.hf_token:
                    self.pipeline = Pipeline.from_pretrained(
                        self.model_name,
                        use_auth_token=self.hf_token
                    )
                else:
                    self.pipeline = Pipeline.from_pretrained(self.model_name)
            
            # 将pipeline移到设备上
            if self.device.type == "cuda":
                self.pipeline.to(self.device)
            
            print(f"说话人计数器初始化完成")
            print(f"  模型: {self.local_model_path or self.model_name}")
            
        except ImportError as e:
            raise ImportError(
                "请先安装pyannote.audio: pip install pyannote.audio\n"
                f"错误详情: {e}"
            )
        except Exception as e:
            error_msg = str(e)
            if "401" in error_msg or "access" in error_msg.lower():
                raise ValueError(
                    "需要Hugging Face Token才能使用此模型。\n"
                    "请按以下步骤操作：\n"
                    "1. 注册Hugging Face账号: https://huggingface.co\n"
                    "2. 访问 https://huggingface.co/pyannote/speaker-diarization-3.1 点击 'Agree'\n"
                    "3. 访问 https://huggingface.co/pyannote/segmentation-3.0 点击 'Agree'\n"
                    "4. 创建Token: https://huggingface.co/settings/tokens\n"
                    "5. 使用 SpeakerCounter(hf_token='your_token') 初始化"
                )
            raise
    
    def count_speakers(self, 
                        audio_path: str,
                        num_speakers: Optional[int] = None,
                        min_speakers: Optional[int] = None,
                        max_speakers: Optional[int] = None) -> Dict:
        """
        统计音频中的说话人数量
        
        Args:
            audio_path: 音频文件路径
            num_speakers: 已知的说话人数量（可选，提供可提高准确率）
            min_speakers: 最小说话人数量（可选）
            max_speakers: 最大说话人数量（可选）
            
        Returns:
            dict: 包含说话人数量和详细信息的字典
        """
        if self.pipeline is None:
            raise RuntimeError("Pipeline未初始化")
        
        diarization_kwargs = {}
        if num_speakers is not None:
            diarization_kwargs["num_speakers"] = num_speakers
        if min_speakers is not None:
            diarization_kwargs["min_speakers"] = min_speakers
        if max_speakers is not None:
            diarization_kwargs["max_speakers"] = max_speakers
        
        diarization = self.pipeline(audio_path, **diarization_kwargs)
        
        speaker_labels = set(diarization.labels())
        num_speakers_detected = len(speaker_labels)
        
        speaker_timeline = self._get_speaker_timeline(diarization)
        speaker_durations = self._calculate_speaker_durations(diarization)
        audio_duration = diarization.get_timeline().extent().end
        
        result = {
            'num_speakers': num_speakers_detected,
            'speaker_labels': list(speaker_labels),
            'audio_duration': audio_duration,
            'speaker_durations': speaker_durations,
            'speaker_timeline': speaker_timeline,
            'diarization': diarization
        }
        
        return result
    
    def _get_speaker_timeline(self, diarization) -> List[Dict]:
        """获取说话人时间线"""
        timeline = []
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            timeline.append({
                'start': turn.start,
                'end': turn.end,
                'duration': turn.end - turn.start,
                'speaker': speaker
            })
        return timeline
    
    def _calculate_speaker_durations(self, diarization) -> Dict[str, float]:
        """计算每个说话人的总时长"""
        durations = {}
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            duration = turn.end - turn.start
            if speaker not in durations:
                durations[speaker] = 0.0
            durations[speaker] += duration
        return durations
    
    def get_speaker_segments(self, 
                             audio_path: str,
                             num_speakers: Optional[int] = None) -> Dict[str, List[Tuple[float, float]]]:
        """获取每个说话人的说话片段"""
        result = self.count_speakers(audio_path, num_speakers=num_speakers)
        
        segments = {}
        for item in result['speaker_timeline']:
            speaker = item['speaker']
            if speaker not in segments:
                segments[speaker] = []
            segments[speaker].append((item['start'], item['end']))
        
        return segments
    
    def analyze_speaker_activity(self, 
                                 audio_path: str,
                                 num_speakers: Optional[int] = None) -> Dict:
        """分析说话人活动"""
        result = self.count_speakers(audio_path, num_speakers=num_speakers)
        
        total_duration = result['audio_duration']
        speaker_durations = result['speaker_durations']
        
        speaker_ratios = {
            speaker: duration / total_duration 
            for speaker, duration in speaker_durations.items()
        }
        
        total_speech_time = sum(speaker_durations.values())
        silence_time = total_duration - total_speech_time
        silence_ratio = silence_time / total_duration if total_duration > 0 else 0
        
        main_speaker = max(speaker_durations, key=speaker_durations.get) if speaker_durations else None
        
        timeline = result['speaker_timeline']
        turn_takes = 0
        for i in range(1, len(timeline)):
            if timeline[i]['speaker'] != timeline[i-1]['speaker']:
                turn_takes += 1
        
        analysis = {
            'num_speakers': result['num_speakers'],
            'total_duration': total_duration,
            'total_speech_time': total_speech_time,
            'silence_time': silence_time,
            'silence_ratio': silence_ratio,
            'speaker_durations': speaker_durations,
            'speaker_ratios': speaker_ratios,
            'main_speaker': main_speaker,
            'main_speaker_ratio': speaker_ratios.get(main_speaker, 0),
            'num_turn_takes': turn_takes,
            'avg_turn_duration': total_speech_time / (turn_takes + 1) if turn_takes > 0 else total_speech_time
        }
        
        return analysis
    
    def print_summary(self, result: Dict):
        """打印分析结果摘要"""
        print("\n" + "="*50)
        print("说话人分析结果")
        print("="*50)
        print(f"检测到的说话人数量: {result['num_speakers']}")
        print(f"音频总时长: {result['audio_duration']:.2f} 秒")
        
        if 'speaker_durations' in result:
            print("\n各说话人说话时长:")
            for speaker, duration in sorted(result['speaker_durations'].items()):
                if 'speaker_ratios' in result:
                    ratio = result['speaker_ratios'][speaker]
                    print(f"  {speaker}: {duration:.2f} 秒 ({ratio*100:.1f}%)")
                else:
                    print(f"  {speaker}: {duration:.2f} 秒")
        
        if 'num_turn_takes' in result:
            print(f"\n说话人交替次数: {result['num_turn_takes']}")
            print(f"主要说话人: {result['main_speaker']} ({result['main_speaker_ratio']*100:.1f}%)")
            print(f"静音比例: {result['silence_ratio']*100:.1f}%")
        
        print("="*50)


def quick_count_speakers(audio_path: str, 
                         hf_token: Optional[str] = None,
                         num_speakers: Optional[int] = None,
                         local_model_path: Optional[str] = None) -> int:
    """
    快速统计说话人数量（便捷函数）
    
    Args:
        audio_path: 音频文件路径
        hf_token: Hugging Face Token（在线模式）
        num_speakers: 已知的说话人数量（可选）
        local_model_path: 本地模型路径（离线模式）
        
    Returns:
        int: 检测到的说话人数量
    """
    counter = SpeakerCounter(
        hf_token=hf_token,
        local_model_path=local_model_path
    )
    result = counter.count_speakers(audio_path, num_speakers=num_speakers)
    return result['num_speakers']


def download_models_for_offline(
        hf_token: str,
        output_dir: str = "models",
        model_names: List[str] = None):
    """
    下载模型用于离线部署
    
    Args:
        hf_token: Hugging Face访问令牌
        output_dir: 输出目录
        model_names: 要下载的模型名称列表
    
    Example:
        download_models_for_offline("your_hf_token")
    """
    try:
        from huggingface_hub import snapshot_download
        
        if model_names is None:
            model_names = [
                "pyannote/speaker-diarization-3.1",
                "pyannote/segmentation-3.0",
                "pyannote/wespeaker-voxceleb-resnet34-LM"
            ]
        
        print(f"开始下载 {len(model_names)} 个模型...")
        
        for model_name in model_names:
            local_dir = os.path.join(output_dir, model_name.split("/")[-1])
            print(f"\n下载 {model_name} -> {local_dir}")
            
            snapshot_download(
                repo_id=model_name,
                local_dir=local_dir,
                token=hf_token
            )
            
            print(f"✓ 下载完成")
        
        print(f"\n所有模型已下载到 {output_dir}")
        
    except ImportError:
        raise ImportError("请安装 huggingface-hub: pip install huggingface-hub")
    except Exception as e:
        raise RuntimeError(f"下载失败: {e}")