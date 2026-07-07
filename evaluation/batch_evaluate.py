"""
VoxConverse 批量评估脚本

同时对 ECAPA-TDNN 和 XVector 两个模型跑 216 段音频，
计算 DER、帧准确率、说话人数准确率等指标。
"""
import os
import sys
import time
import json
import numpy as np
import librosa

# 确保能找到各模块
# evaluation/ 在 SoundCUDA-main/SoundCUDA-main/ 下，而音频在上一级 Parent/ 下
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PARENT_DIR = os.path.dirname(PROJECT_DIR)
sys.path.insert(0, PROJECT_DIR)
sys.path.insert(0, os.path.join(PROJECT_DIR, 'ECAPA-TDNN'))
sys.path.insert(0, os.path.join(PROJECT_DIR, 'ModelTrain'))

from evaluation.rttm_parser import parse_rttm, rttm_to_frame_labels
from evaluation.diarization_eval import evaluate_one

# 配置路径（数据在 PARENT_DIR 下，代码在 PROJECT_DIR 下）
AUDIO_DIR = os.path.join(PARENT_DIR, 'voxconverse_dev_wav', 'audio')
RTTM_DIR = os.path.join(PARENT_DIR, 'labels', 'dev')

# ECAPA 参数
ECAPA_SEG_DUR = 1.0
ECAPA_OVERLAP = 0.5

# XVector 参数
XVECTOR_SEG_DUR = 0.5
XVECTOR_OVERLAP = 0.5


def get_audio_duration(audio_path):
    """获取音频时长（秒）"""
    try:
        y, sr = librosa.load(audio_path, sr=None, mono=True)
        return len(y) / sr
    except:
        return 0


def run_ecapa(audio_path):
    """运行 ECAPA-TDNN 模型，返回帧级标签"""
    try:
        from ecapa_speaker import analyze_speakers
        result = analyze_speakers(audio_path, seg_dur=ECAPA_SEG_DUR, overlap=ECAPA_OVERLAP)
        return result
    except Exception as e:
        return {'error': str(e), 'timeline': []}


def run_xvector(audio_path):
    """运行 XVector 模型，返回帧级标签"""
    try:
        from speaker_timeline import analyze_speakers
        result = analyze_speakers(audio_path, seg_dur=XVECTOR_SEG_DUR, overlap=XVECTOR_OVERLAP)
        return result
    except Exception as e:
        return {'error': str(e), 'timeline': []}


def model_timeline_to_labels(timeline, n_frames, model_name):
    """
    将模型输出的 timeline [(start_time, speaker_id), ...] 转换为帧级标签。
    对于 ECAPA，帧长=1.0 重叠=0.5，hop=0.5s
    对于 XVector，帧长=0.5 重叠=0.5，hop=0.25s
    """
    if model_name == 'ecapa':
        hop = ECAPA_SEG_DUR * (1 - ECAPA_OVERLAP)  # 0.5s
    else:
        hop = XVECTOR_SEG_DUR * (1 - XVECTOR_OVERLAP)  # 0.25s

    labels = [-1] * n_frames
    for i, (start, spk_id) in enumerate(timeline):
        if i < n_frames and spk_id >= 0:
            labels[i] = int(spk_id)
    return labels


# 累积错误信息
_first_errors = {}


def evaluate_model(model_name, run_fn, audio_files, seg_dur, overlap):
    """批量评估一个模型的全部音频"""
    hop = seg_dur * (1 - overlap)
    results = []
    errors = 0

    for i, audio_file in enumerate(audio_files):
        file_id = os.path.splitext(os.path.basename(audio_file))[0]
        rttm_path = os.path.join(RTTM_DIR, f'{file_id}.rttm')

        if not os.path.exists(rttm_path):
            continue

        dur = get_audio_duration(audio_file)
        if dur == 0:
            continue

        # RTTM 帧标签
        n_frames = max(1, int(dur / hop) + 1)
        ref_labels, ref_n_spk = rttm_to_frame_labels(rttm_path, seg_dur, overlap, dur)

        # 运行模型
        model_result = run_fn(audio_file)

        if 'error' in model_result:
            errors += 1
            # 记录前 3 个错误，方便排查
            key = f"{model_name}_{file_id}"
            if len(_first_errors) < 3 and key not in _first_errors:
                _first_errors[key] = model_result['error']
            # 每 5 个错误打印一次进度
            if errors % 5 == 0:
                print(f"  [{model_name.upper()}] 错误: {errors}, 最近: {file_id}", flush=True)
            continue

        # 模型帧标签
        hyp_labels = model_timeline_to_labels(
            model_result['timeline'],
            n_frames, model_name
        )

        # 评估
        eval_result = evaluate_one(ref_labels, hyp_labels, ref_n_spk)
        eval_result['file_id'] = file_id
        eval_result['duration'] = round(dur, 1)
        eval_result['model'] = model_name
        results.append(eval_result)

        # 逐文件打印进度
        line = f"  [{model_name.upper()}] {len(results)}/{len(audio_files)}  {file_id} dur={dur:.0f}s ref={ref_n_spk}spk DER={eval_result['DER']:.1f}%"
        print(line, flush=True)

        if (i + 1) % 20 == 0:
            print(f"  [{model_name.upper()}] {i+1}/{len(audio_files)} 完成...")

    return results, errors


def print_summary(name, results):
    """打印汇总指标"""
    if not results:
        print(f"\n{'='*60}")
        print(f"  {name}: 无有效结果")
        print(f"{'='*60}")
        return

    der_list = [r['DER'] for r in results]
    acc_list = [r['frame_accuracy'] for r in results]
    spk_correct = sum(1 for r in results if r['n_spk_correct'])
    total_miss = sum(r['miss'] for r in results)
    total_fa = sum(r['false_alarm'] for r in results)
    total_conf = sum(r['speaker_confusion'] for r in results)
    total_voice = sum(r['voice_frames'] for r in results)
    overall_der = (total_miss + total_fa + total_conf) / total_voice * 100 if total_voice > 0 else 0

    print(f"\n{'='*60}")
    print(f"  {name} — {len(results)} 段音频评估结果")
    print(f"{'='*60}")
    print(f"  总体 DER:              {overall_der:.2f}%")
    print(f"  DER 平均值:            {np.mean(der_list):.2f}%")
    print(f"  DER 中位数:            {np.median(der_list):.2f}%")
    print(f"  DER 标准差:            {np.std(der_list):.2f}%")
    print(f"  帧级准确率 均值:       {np.mean(acc_list):.2f}%")
    print(f"  说话人数正确率:        {spk_correct}/{len(results)} ({spk_correct/len(results)*100:.1f}%)")
    print(f"  其中 —")
    print(f"    漏检 (Miss):         {total_miss} 帧")
    print(f"    虚警 (False Alarm):  {total_fa} 帧")
    print(f"    说话人混淆:          {total_conf} 帧")
    print(f"    有声帧总数:          {total_voice}")

    # 按说话人数分组统计
    by_spk = {}
    for r in results:
        n = r['ref_n_speakers']
        if n not in by_spk:
            by_spk[n] = []
        by_spk[n].append(r)

    print(f"\n  按说话人数分组 DER:")
    for n_spk in sorted(by_spk.keys()):
        group = by_spk[n_spk]
        avg_der = np.mean([r['DER'] for r in group])
        print(f"    {n_spk}人:  {len(group)}段, 平均 DER={avg_der:.2f}%")


def main():
    print("=" * 60)
    print("  EchoSense 说话人日志评估 — VoxConverse Dev 集")
    print("=" * 60)

    # 扫描音频文件
    audio_files = sorted([
        os.path.join(AUDIO_DIR, f) for f in os.listdir(AUDIO_DIR) if f.endswith('.wav')
    ])

    rttm_files = set(
        f.replace('.rttm', '') for f in os.listdir(RTTM_DIR) if f.endswith('.rttm')
    )

    # 只取 RTTM 和 WAV 交集
    valid = [af for af in audio_files if os.path.splitext(os.path.basename(af))[0] in rttm_files]
    print(f"\n音频文件: {len(audio_files)}, RTTM 文件: {len(rttm_files)}, 配对成功: {len(valid)}")

    # ===== 评估 XVector =====
    print(f"\n[1/2] 评估 XVector 模型...")
    xv_results, xv_errors = evaluate_model(
        'xvector', run_xvector, valid, XVECTOR_SEG_DUR, XVECTOR_OVERLAP
    )
    print(f"  完成: {len(xv_results)} 段, 错误: {xv_errors}")

    # ===== 评估 ECAPA =====
    print(f"\n[2/2] 评估 ECAPA-TDNN 模型...")
    ec_results, ec_errors = evaluate_model(
        'ecapa', run_ecapa, valid, ECAPA_SEG_DUR, ECAPA_OVERLAP
    )
    print(f"  完成: {len(ec_results)} 段, 错误: {ec_errors}")

    # ===== 打印错误详情 =====
    if _first_errors:
        print(f"\n  错误详情（前3个）:")
        for key, err in _first_errors.items():
            print(f"    [{key}] {err}")

    # ===== 汇总 =====
    print_summary("XVector (自训练模型)", xv_results)
    print_summary("ECAPA-TDNN (预训练模型)", ec_results)

    # ===== 保存详细结果 =====
    output = {
        'evaluation_date': time.strftime('%Y-%m-%d %H:%M:%S'),
        'total_audio': len(valid),
        'xvector': {
            'results': xv_results,
            'errors': xv_errors,
        },
        'ecapa': {
            'results': ec_results,
            'errors': ec_errors,
        },
    }
    output_path = os.path.join(PROJECT_DIR, 'evaluation', 'eval_results.json')
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n详细结果已保存至: {output_path}")

    # ===== 对比结论 =====
    print(f"\n{'='*60}")
    print(f"  对比总结")
    print(f"{'='*60}")
    if xv_results and ec_results:
        xv_der = np.mean([r['DER'] for r in xv_results])
        ec_der = np.mean([r['DER'] for r in ec_results])
        xv_acc = np.mean([r['frame_accuracy'] for r in xv_results])
        ec_acc = np.mean([r['frame_accuracy'] for r in ec_results])
        print(f"              XVector        ECAPA-TDNN")
        print(f"  DER 均值:   {xv_der:5.2f}%         {ec_der:5.2f}%")
        print(f"  帧准确率:   {xv_acc:5.2f}%         {ec_acc:5.2f}%")
        print(f"  完成数:     {len(xv_results):>4}            {len(ec_results):>4}")
        if ec_der < xv_der:
            print(f"\n  ECAPA-TDNN 优于 XVector (DER 低 {xv_der - ec_der:.2f}%)")
        else:
            print(f"\n  XVector 优于 ECAPA-TDNN (DER 低 {ec_der - xv_der:.2f}%)")


if __name__ == '__main__':
    main()
