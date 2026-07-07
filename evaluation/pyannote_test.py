#!/usr/bin/env python3
"""pyannote 3 文件快速验证 — 绕过 torchcodec，用 soundfile 预加载"""
import os, sys, time

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE = os.path.dirname(PROJECT)
sys.path.insert(0, PROJECT)
sys.path.insert(0, os.path.join(PROJECT, 'evaluation'))

import numpy as np
import soundfile as sf
import torch
from pyannote.audio import Pipeline

from rttm_parser import parse_rttm, rttm_to_frame_labels
from diarization_eval import evaluate_one

AUDIO = os.path.join(BASE, 'voxconverse_dev_wav', 'audio')
LABELS = os.path.join(BASE, 'labels', 'dev')
TEST_FILES = ['hqyok', 'tucrg', 'tfvyr']  # ~25s, 1-3 人
SEG_DUR, OVERLAP = 0.5, 0.25

print("=" * 60)
print("  pyannote 快速测试 — 3 个短文件")
print("=" * 60)

print("\n[pyannote] 加载模型...")
pipeline = Pipeline.from_pretrained('pyannote/speaker-diarization-3.1', token='YOUR_HF_TOKEN')  # 请替换为你的 HuggingFace token
pipeline.to(torch.device('cpu'))
print("[pyannote] 加载完成\n")

results = []
for fid in TEST_FILES:
    wav = os.path.join(AUDIO, f'{fid}.wav')
    rttm = os.path.join(LABELS, f'{fid}.rttm')
    if not os.path.exists(wav):
        print(f"  {fid}: 文件不存在")
        continue

    y, sr = sf.read(wav, dtype='float32')
    if y.ndim > 1:
        y = y.mean(axis=1)
    dur = len(y) / sr
    n_frames = int(dur / (SEG_DUR * (1 - OVERLAP))) + 1

    # 绕过 torchcodec：预加载音频
    audio_dict = {'waveform': torch.from_numpy(y).unsqueeze(0), 'sample_rate': sr}

    t0 = time.time()
    dia = pipeline(audio_dict)
    elapsed = time.time() - t0

    # pyannote 4.x: speaker_diarization 就是 Annotation
    annotation = dia.speaker_diarization

    hyp_labels = np.full(n_frames, -1, dtype=int)
    spk_map = {}
    next_spk = 0
    n_turns = 0
    for turn, _, speaker in annotation.itertracks(yield_label=True):
        n_turns += 1
        fs = int(turn.start / (SEG_DUR * (1 - OVERLAP)))
        fe = int(turn.end / (SEG_DUR * (1 - OVERLAP))) + 1
        if speaker not in spk_map:
            spk_map[speaker] = next_spk
            next_spk += 1
        hyp_labels[max(0, fs):min(n_frames, fe)] = spk_map[speaker]

    ref_labels, ref_n = rttm_to_frame_labels(rttm, SEG_DUR, OVERLAP, dur)
    ev = evaluate_one(ref_labels, hyp_labels, ref_n)

    print(f"--- {fid} (dur={dur:.0f}s) ---")
    print(f"  pyannote: {n_turns} 段, {next_spk} 人, {elapsed:.1f}s")
    print(f"  DER={ev['DER']:.1f}%  acc={ev['frame_accuracy']:.1f}%  人数正确={ev['n_spk_correct']}")
    results.append({'file_id': fid, 'DER': ev['DER'], 'frame_accuracy': ev['frame_accuracy'],
                     'n_spk_correct': ev['n_spk_correct'], 'elapsed_s': elapsed})

print("\n" + "=" * 60)
if results:
    avg_der = sum(r['DER'] for r in results) / len(results)
    avg_acc = sum(r['frame_accuracy'] for r in results) / len(results)
    n_ok = sum(1 for r in results if r['n_spk_correct'])
    print(f"  平均 DER: {avg_der:.1f}%  帧准确率: {avg_acc:.1f}%  人数正确: {n_ok}/{len(results)}")
print("=" * 60)
