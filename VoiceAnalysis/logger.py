"""
VoiceAnalysis 日志配置模块
"""

import os
import logging
from datetime import datetime
from typing import Optional


def setup_logger(name: str, log_dir: str = None) -> logging.Logger:
    """
    设置并返回一个配置好的 logger
    
    Args:
        name: logger 名称
        log_dir: 日志存储目录，默认使用项目根目录下的 logs 文件夹
    
    Returns:
        配置好的 logger 对象
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    
    if logger.handlers:
        logger.handlers.clear()
    
    if log_dir is None:
        log_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'logs')
    
    os.makedirs(log_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d')
    log_file = os.path.join(log_dir, f'{name}_{timestamp}.log')
    
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    logger.debug(f"日志系统初始化完成，日志文件: {log_file}")
    
    return logger


def log_analysis_start(logger: logging.Logger, audio_file: str) -> None:
    """
    记录分析开始
    
    Args:
        logger: 日志记录器
        audio_file: 音频文件路径
    """
    logger.info("=" * 60)
    logger.info("人声分析任务开始")
    logger.info("=" * 60)
    logger.info(f"音频文件: {audio_file}")


def log_analysis_complete(logger: logging.Logger, result: dict) -> None:
    """
    记录分析完成
    
    Args:
        logger: 日志记录器
        result: 分析结果
    """
    logger.info("人声分析任务完成")
    logger.info(f"分析时长: {result.get('duration', 0):.2f}秒")
    logger.info(f"分析段数: {result.get('num_segments', 0)}")
    logger.info(f"总耗时: {result.get('analysis_time', 0):.2f}秒")
    
    emotion_result = result.get('emotion_result', {})
    dominant_emotion = emotion_result.get('dominant_emotion', 'unknown')
    confidence = emotion_result.get('confidence', 0)
    logger.info(f"主导情绪: {dominant_emotion}, 置信度: {confidence:.2%}")
    
    logger.info("=" * 60)


def log_error(logger: logging.Logger, error: Exception, context: str = "") -> None:
    """
    记录错误信息
    
    Args:
        logger: 日志记录器
        error: 异常对象
        context: 错误上下文
    """
    logger.error(f"发生错误: {context}")
    logger.error(f"错误类型: {type(error).__name__}")
    logger.error(f"错误详情: {str(error)}", exc_info=True)


def log_performance(logger: logging.Logger, phase: str, duration: float) -> None:
    """
    记录性能信息
    
    Args:
        logger: 日志记录器
        phase: 阶段名称
        duration: 耗时（秒）
    """
    logger.debug(f"[{phase}] 耗时: {duration:.4f}秒")


def log_features(logger: logging.Logger, features: dict) -> None:
    """
    记录特征提取摘要
    
    Args:
        logger: 日志记录器
        features: 特征字典
    """
    logger.info("特征提取摘要:")
    
    f0_features = features.get('f0', {})
    if f0_features:
        logger.info(f"  基频特征: 平均={f0_features.get('mean_f0_mean', 0):.1f}Hz, "
                    f"范围={f0_features.get('range_f0_mean', 0):.1f}Hz")
    
    spectral_features = features.get('spectral', {})
    if spectral_features:
        logger.info(f"  频谱特征: 质心={spectral_features.get('spectral_centroid_mean_mean', 0):.1f}Hz")
    
    energy_features = features.get('energy', {})
    if energy_features:
        logger.info(f"  能量特征: RMS={energy_features.get('rms_mean_mean', 0):.4f}")


def log_emotion_distribution(logger: logging.Logger, distribution: dict) -> None:
    """
    记录情绪分布
    
    Args:
        logger: 日志记录器
        distribution: 情绪分布字典
    """
    logger.info("情绪分布:")
    
    emotion_names_cn = {
        'neutral': '中性',
        'happy': '快乐',
        'sad': '悲伤',
        'angry': '愤怒',
        'fear': '恐惧',
        'surprise': '惊讶',
        'disgust': '厌恶'
    }
    
    for emotion, ratio in sorted(distribution.items(), key=lambda x: x[1], reverse=True):
        emotion_cn = emotion_names_cn.get(emotion, emotion)
        logger.info(f"  {emotion_cn}: {ratio:.1%}")


def log_gui_operation(logger: logging.Logger, operation: str, details: str = "") -> None:
    """
    记录GUI操作
    
    Args:
        logger: 日志记录器
        operation: 操作名称
        details: 操作详情
    """
    logger.info(f"[GUI] {operation}")
    if details:
        logger.debug(f"[GUI详情] {details}")


def log_parallel_info(logger: logging.Logger, use_parallel: bool, workers: int) -> None:
    """
    记录并行处理信息
    
    Args:
        logger: 日志记录器
        use_parallel: 是否启用并行
        workers: 工作进程数
    """
    if use_parallel:
        logger.info(f"并行处理已启用，工作进程数: {workers}")
    else:
        logger.info("使用串行处理模式")
