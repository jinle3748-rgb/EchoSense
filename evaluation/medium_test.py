#!/usr/bin/env python3
"""中等长度文件测试：60-120s 音频，仅 ECAPA-TDNN"""

import os
import sys
import json
import time
import numpy as np

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PARENT_DIR = os.path.dirname(PROJECT_DIR)
sys.path.insert(0, PROJECT_DIR)
sys.path.insert(0, os.path.join(PROJECT_DIR, 'ECAPA-TDNN'))

from evaluation.rttm_parser import parse_rttm, rttm_to_frame_labels
from evaluation.diarization_eval import evaluate_one
from ecapa_speaker import analyze_speakers

AUDIO_DIR = os.path.join(PARENT_DIR, 'voxconverse_dev_wav', 'audio')
RTTM_DIR = os.path.join(PARENT_DIR, 'labels', 'dev')

# 25 个 60-120s 文件
TEST_FILES = [
    'bkwns', 'abjxc', 'syiwe', 'qppll', 'cobal', 'oenox', 'bwzyf',
    'jiqvr', 'jyirt', 'hiyis', 'plbbw', 'vysqj', 'sikkm', 'wjhgf',
    'lknjp', 'mevkw', 'kctgl', 'zfkap', 'iqtde', 'xiglo', 'jsmbi',
    'qydmg', 'akthc', 'exymw', 'kbkon',
]

SEG_DUR = 1.0
OVERLAP = 0.5
HOP = SEG_DUR * (1 - OVERLAP)


def model_timeline_to_labels(timeline, n_frames):
    """timeline [(start_time, speaker_id), ...] → 帧级标签"""
    labels = [-1] * n_frames
    for entry in timeline:
        start_time, speaker_id = entry[0], int(entry[1])
        if speaker_id < 0:
            continue
        start_frame = max(0, int(start_time / HOP))
        end_frame = min(n_frames, int((start_time + SEG_DUR) / HOP) + 1)
        for f in range(start_frame, end_frame):
            if labels[f] == -1:
                labels[f] = speaker_id
            elif labels[f] != speaker_id:
                labels[f] = -2
    labels = [l if l >= 0 else -1 for l in labels]
    return labels


def main():
    print("=" * 60)
    print("  EchoSense 中等长度测试 — 25 文件 (60-120s)")
    print("  ECAPA-TDNN 模型")
    print("=" * 60)

    total_start = time.time()
    all_results = []
    errors = 0

    for i, fid in enumerate(TEST_FILES):
        wav = os.path.join(AUDIO_DIR, f'{fid}.wav')
        rttm = os.path.join(RTTM_DIR, f'{fid}.rttm')

        if not os.path.exists(wav):
            print(f"  [SKIP] {fid} — WAV 不存在")
            continue
        if not os.path.exists(rttm):
            print(f"  [SKIP] {fid} — RTTM 不存在")
            continue

        # 文件信息
        segs = parse_rttm(rttm)
        ref_n_spk = len(set(s[2] for s in segs))
        n_frames = max(1, int(sum(s[1] - s[0] for s in segs) / HOP + len(segs) * SEG_DUR / HOP))
        # 用 RTTM 最后一帧 + buffer 估算
        max_end = max(s[1] for s in segs)
        n_frames = max(1, int(max_end / HOP) + 2)

        # RTTM → 帧标签
        ref_labels, _ = rttm_to_frame_labels(rttm, SEG_DUR, OVERLAP, max_end)
        n_voice = sum(1 for l in ref_labels if l >= 0)

        # ECAPA 推理
        t0 = time.time()
        try:
            result = analyze_speakers(wav, seg_dur=SEG_DUR, overlap=OVERLAP)
        except Exception as e:
            errors += 1
            print(f"  [{i+1}/25] {fid}  ERROR: {e}", flush=True)
            continue

        elapsed = time.time() - t0
        pred_n_spk = result['speaker_count']

        # 模型帧标签
        hyp_labels = model_timeline_to_labels(result['timeline'], n_frames)

        # 评估
        eval_result = evaluate_one(ref_labels, hyp_labels, ref_n_spk)
        eval_result['file_id'] = fid
        eval_result['duration'] = round(max_end, 1)
        eval_result['model'] = 'ecapa'
        eval_result['ref_n_speakers'] = ref_n_spk
        eval_result['pred_n_speakers'] = pred_n_spk
        eval_result['elapsed_s'] = round(elapsed, 1)
        all_results.append(eval_result)

        print(f"  [{i+1}/25] {fid}  {max_end:.0f}s  ref={ref_n_spk}spk pred={pred_n_spk}spk  "
              f"DER={eval_result['DER']:.1f}%  acc={eval_result['frame_accuracy']:.1f}%  "
              f"{elapsed:.0f}s",
              flush=True)

    total_time = time.time() - total_start

    # ===== 汇总 =====
    if not all_results:
        print("\n[FAIL] 无有效结果")
        return

    der_list = [r['DER'] for r in all_results]
    acc_list = [r['frame_accuracy'] for r in all_results]
    spk_correct = sum(1 for r in all_results if r['n_spk_correct'])
    total_miss = sum(r['miss'] for r in all_results)
    total_fa = sum(r['false_alarm'] for r in all_results)
    total_conf = sum(r['speaker_confusion'] for r in all_results)
    total_voice = sum(r['voice_frames'] for r in all_results)
    overall_der = (total_miss + total_fa + total_conf) / total_voice * 100 if total_voice > 0 else 0

    print("\n" + "=" * 60)
    print(f"  汇总 ({len(all_results)} 文件, 总耗时 {total_time:.0f}s)")
    print("=" * 60)
    print(f"  总体 DER:              {overall_der:.2f}%")
    print(f"  DER 平均值:            {np.mean(der_list):.2f}%")
    print(f"  DER 中位数:            {np.median(der_list):.2f}%")
    print(f"  DER 标准差:            {np.std(der_list):.2f}%")
    print(f"  帧级准确率 均值:       {np.mean(acc_list):.2f}%")
    print(f"  说话人数正确率:        {spk_correct}/{len(all_results)} ({spk_correct/len(all_results)*100:.1f}%)")
    print(f"  错误数:                {errors}")
    print(f"  其中 —")
    print(f"    漏检 (Miss):         {total_miss} 帧")
    print(f"    虚警 (False Alarm):  {total_fa} 帧")
    print(f"    说话人混淆:          {total_conf} 帧")
    print(f"    有声帧总数:          {total_voice}")

    # 按说话人数分组
    by_spk = {}
    for r in all_results:
        n = r['ref_n_speakers']
        if n not in by_spk:
            by_spk[n] = []
        by_spk[n].append(r)

    print(f"\n  按说话人数分组 DER:")
    for n_spk in sorted(by_spk.keys()):
        group = by_spk[n_spk]
        g_der = np.mean([r['DER'] for r in group])
        g_acc = np.mean([r['frame_accuracy'] for r in group])
        g_correct = sum(1 for r in group if r['n_spk_correct'])
        print(f"    {n_spk}人: {len(group)}段, DER={g_der:.2f}%, "
              f"帧准确率={g_acc:.2f}%, 人数正确={g_correct}/{len(group)}")

    # 保存
    output_path = os.path.join(PROJECT_DIR, 'evaluation', 'medium_test_results.json')
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n结果已保存到: {output_path}")


if __name__ == '__main__':
    main()
