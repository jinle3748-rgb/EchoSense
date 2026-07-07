"""
RTTM 文件解析器 — 读取 VoxConverse 格式的标注文件
"""
import os


def parse_rttm(rttm_path):
    """
    解析单个 RTTM 文件，返回说话人时间段列表。

    RTTM 格式:
        SPEAKER {file_id} 1 {start} {duration} <NA> <NA> {speaker_id} <NA> <NA>

    返回:
        [(start, end, speaker_id), ...]  按 start 排序
    """
    segments = []
    if not os.path.exists(rttm_path):
        return segments

    with open(rttm_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            if parts[0] != 'SPEAKER':
                continue
            # parts: [SPEAKER, file_id, channel, start, duration, <NA>, <NA>, speaker_id, <NA>, <NA>]
            start = float(parts[3])
            duration = float(parts[4])
            speaker_id = parts[7]
            segments.append((start, start + duration, speaker_id))

    # 按时间排序
    segments.sort(key=lambda x: x[0])
    return segments


def rttm_to_frame_labels(rttm_path, seg_dur, overlap, total_dur):
    """
    将 RTTM 标注转换为逐帧标签，帧长与模型输出对齐。

    参数:
        rttm_path: RTTM 文件路径
        seg_dur:   模型使用的片段时长 (秒)
        overlap:   模型使用的重叠比例
        total_dur: 音频总时长 (秒)

    返回:
        list[int or -1]: 每帧的主说话人 ID（整数），无说话人为 -1
        int:             RTTM 中的总说话人数
    """
    segments = parse_rttm(rttm_path)
    hop_len = seg_dur * (1 - overlap)

    # 收集所有 speaker ID，映射为整数
    all_speakers = sorted(set(s[2] for s in segments))
    spk_to_int = {spk: i for i, spk in enumerate(all_speakers)}
    n_speakers = len(all_speakers)

    # 计算帧数
    n_frames = max(1, int(total_dur / hop_len) + 1)
    frame_labels = [-1] * n_frames  # -1 表示静音

    # 为每帧确定主说话人
    for f in range(n_frames):
        frame_start = f * hop_len
        frame_end = min(frame_start + seg_dur, total_dur)

        # 统计该帧内各说话人的时长
        spk_dur = {}
        for start, end, spk in segments:
            overlap_start = max(start, frame_start)
            overlap_end = min(end, frame_end)
            if overlap_end > overlap_start:
                spk_dur[spk] = spk_dur.get(spk, 0) + (overlap_end - overlap_start)

        if spk_dur:
            # 取占该帧时长最多的说话人
            dominant = max(spk_dur, key=spk_dur.get)
            frame_labels[f] = spk_to_int[dominant]

    return frame_labels, n_speakers
