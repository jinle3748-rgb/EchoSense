#!/usr/bin/env python3
"""快速测试：5 个短文件（<45秒）+ ECAPA-TDNN"""

import os
import sys
import json
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), 'ECAPA-TDNN'))

from evaluation.rttm_parser import parse_rttm, rttm_to_frame_labels
from evaluation.diarization_eval import evaluate_one
from ecapa_speaker import analyze_speakers

base = os.path.dirname(os.path.dirname(__file__))

# 最短的 5 个文件（全部 <45 秒）
AUDIO_DIR = os.path.join(os.path.dirname(base), 'voxconverse_dev_wav', 'audio')
RTTM_DIR = os.path.join(os.path.dirname(base), 'labels', 'dev')
TEST_FILES = ['hqyok', 'tucrg', 'tfvyr', 'qrzjk', 'qpylu']

# ECAPA 参数
SEG_DUR = 1.0
OVERLAP = 0.5
HOP = SEG_DUR * (1 - OVERLAP)  # 0.5s

import soundfile as sf


def model_timeline_to_labels(timeline, n_frames, model_name):
    """模型 timeline → 帧级标签。timeline 格式: [(time, speaker_id), ...]"""
    labels = [-1] * n_frames
    seg_dur = 1.0  # ECAPA 默认 1s

    for entry in timeline:
        start_time, speaker_id = entry[0], int(entry[1])
        if speaker_id < 0:
            continue  # 静音

        start_frame = max(0, int(start_time / HOP))
        end_frame = min(n_frames, int((start_time + seg_dur) / HOP) + 1)
        for f in range(start_frame, end_frame):
            if labels[f] == -1:
                labels[f] = speaker_id
            elif labels[f] != speaker_id:
                labels[f] = -2  # 重叠冲突

    # 移除重叠帧
    labels = [l if l >= 0 else -1 for l in labels]
    return labels


def main():
    print("=" * 60)
    print("  EchoSense 快速测试 — 5 个短文件 (ECAPA-TDNN)")
    print("=" * 60)

    total_start = time.time()
    all_results = []

    for fid in TEST_FILES:
        wav = os.path.join(AUDIO_DIR, f'{fid}.wav')
        rttm = os.path.join(RTTM_DIR, f'{fid}.rttm')

        # RTTM 信息
        segs = parse_rttm(rttm)
        info = sf.info(wav)
        dur = info.duration
        ref_n_spk = len(set(s[2] for s in segs))

        print(f"\n--- {fid} ({dur:.0f}s, {ref_n_spk} 说话人) ---", flush=True)

        # RTTM → 帧标签
        n_frames = max(1, int(dur / HOP) + 1)
        ref_labels, _ = rttm_to_frame_labels(rttm, SEG_DUR, OVERLAP, dur)
        n_voice = sum(1 for l in ref_labels if l >= 0)
        print(f"  RTTM: {len(ref_labels)} 帧, {n_voice} 有声 ({n_voice/len(ref_labels)*100:.1f}%)")

        # ECAPA 推理
        t0 = time.time()
        try:
            result = analyze_speakers(wav, seg_dur=SEG_DUR, overlap=OVERLAP)
        except Exception as e:
            print(f"  [ERROR] {e}")
            continue

        elapsed = time.time() - t0
        print(f"  ECAPA: {len(result['timeline'])} 段, {result['speaker_count']} 说话人, "
              f"耗时 {elapsed:.1f}s")

        # 模型帧标签
        hyp_labels = model_timeline_to_labels(result['timeline'], n_frames, 'ecapa')

        # 评估
        eval_result = evaluate_one(ref_labels, hyp_labels, ref_n_spk)
        eval_result['file_id'] = fid
        eval_result['duration'] = round(dur, 1)
        eval_result['model'] = 'ecapa'
        all_results.append(eval_result)

        print(f"  DER={eval_result['DER']:.1f}%, 帧准确率={eval_result['frame_accuracy']:.1f}%, "
              f"说话人正确={eval_result['n_spk_correct']}")

    # 汇总
    if all_results:
        avg_der = sum(r['DER'] for r in all_results) / len(all_results)
        avg_acc = sum(r['frame_accuracy'] for r in all_results) / len(all_results)
        correct_count = sum(1 for r in all_results if r['n_spk_correct'])
        total_time = time.time() - total_start

        print("\n" + "=" * 60)
        print(f"\n  汇总 ({len(all_results)} 文件, {total_time:.0f}s)")
        print(f"  平均 DER: {avg_der:.1f}%")
        print(f"  平均帧准确率: {avg_acc:.1f}%")
        print(f"  说话人数正确: {correct_count}/{len(all_results)}")
        print("=" * 60)

        # 保存
        out = os.path.join(os.path.dirname(__file__), 'quick_test_results.json')
        with open(out, 'w') as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)
        print(f"\n结果已保存到: {out}")
    else:
        print("\n[FAIL] 所有文件均失败")


if __name__ == '__main__':
    main()
