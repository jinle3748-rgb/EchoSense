"""对比 energy VAD vs silero VAD 对在线追踪的影响 (25文件)"""
import json, os, sys, time
import numpy as np
import torch
sys.path.insert(0, '.')
from rttm_parser import parse_rttm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from blind_count import (
    TEST_25, AUDIO, SR_TARGET, get_embedding_model,
    energy_vad, merge_short_segments, extract_timeline_embeddings,
    online_track
)

RTTM_DIR = r'c:\Users\hp\Desktop\SoundCUDA-main\labels\dev'

# ---- Silero VAD (pip package) ----
from silero_vad import load_silero_vad, get_speech_timestamps
_silero_model = None

def silero_vad_segments(y, sr=16000):
    global _silero_model
    if _silero_model is None:
        _silero_model = load_silero_vad()

    audio_t = torch.from_numpy(y).float()
    ts = get_speech_timestamps(
        audio_t, _silero_model,
        threshold=0.5,
        sampling_rate=sr,
        min_speech_duration_ms=200,
        min_silence_duration_ms=200,
        return_seconds=True
    )
    return [(t['start'], t['end']) for t in ts]


# ---- 对比评测 ----
def run_eval(vad_name, vad_fn, merge_fn, gt):
    import soundfile as sf
    from scipy.signal import resample as scipy_resample

    model = get_embedding_model()
    preds = {}
    segment_counts = {}

    wav_files = [(f, os.path.join(AUDIO, f + '.wav')) for f in TEST_25
                 if os.path.exists(os.path.join(AUDIO, f + '.wav'))]

    for fid, wav in wav_files:
        y, sr_file = sf.read(wav, dtype='float32')
        if y.ndim > 1:
            y = y.mean(axis=1)
        if sr_file != SR_TARGET:
            y = scipy_resample(y, int(len(y) * SR_TARGET / sr_file))

        raw_segs = vad_fn(y, SR_TARGET)
        voice_segs = merge_fn(raw_segs) if merge_fn else raw_segs
        segment_counts[fid] = len(voice_segs)

        if len(voice_segs) == 0:
            preds[fid] = 1
        else:
            timeline = extract_timeline_embeddings(y, SR_TARGET, voice_segs, model)
            if len(timeline) < 2:
                preds[fid] = 1
            else:
                k, _ = online_track(timeline, threshold=0.55, ema=0.25, min_confirm=4)
                preds[fid] = k

    correct = sum(1 for f in preds if f in gt and preds[f] == gt[f])
    mae = sum(abs(preds[f] - gt[f]) for f in preds if f in gt) / len([f for f in preds if f in gt])
    return preds, correct, mae, segment_counts


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    gt = {}
    for fid in TEST_25:
        rttm_path = os.path.join(RTTM_DIR, fid + '.rttm')
        if os.path.exists(rttm_path):
            segs = parse_rttm(rttm_path)
            gt[fid] = len(set(s[2] for s in segs))

    n = len([f for f in TEST_25 if f in gt])

    print("="*75)
    print("  VAD 对比: Energy (当前) vs Silero (神经网络)")
    print("="*75)

    # Energy
    print("\n[1/2] Energy VAD (当前) ...")
    t0 = time.time()
    preds_e, correct_e, mae_e, segs_e = run_eval("Energy", energy_vad, merge_short_segments, gt)
    print(f"  准确率: {correct_e}/{n} = {correct_e/n*100:.1f}%  MAE={mae_e:.3f}  耗时{time.time()-t0:.0f}s")

    # Silero (轻量合并: 只过滤<0.8s的孤立段)
    print("\n[2/2] Silero VAD ...")
    def silero_filter(raw):
        return [(s, e) for s, e in raw if e - s >= 0.8]

    t0 = time.time()
    preds_s, correct_s, mae_s, segs_s = run_eval("Silero", silero_vad_segments, silero_filter, gt)
    print(f"  准确率: {correct_s}/{n} = {correct_s/n*100:.1f}%  MAE={mae_s:.3f}  耗时{time.time()-t0:.0f}s")

    # 逐文件对比
    print(f"\n{'='*75}")
    print(f"  逐文件对比")
    print(f"{'文件':10s}  {'真实':>4s}  {'Energy':>6s}  {'Silero':>6s}  {'E段':>4s}  {'S段':>4s}  {'变化':>6s}")
    print('-'*55)

    for fid in TEST_25:
        if fid not in gt:
            continue
        gt_k = gt[fid]
        pe = preds_e.get(fid, '?')
        ps = preds_s.get(fid, '?')
        se = segs_e.get(fid, 0)
        ss = segs_s.get(fid, 0)
        change = ''
        if isinstance(pe, int) and isinstance(ps, int):
            if pe == gt_k and ps != gt_k:
                change = '变差!'
            elif pe != gt_k and ps == gt_k:
                change = '变好!'
            elif pe != gt_k and ps != gt_k:
                change = '仍错'
            else:
                change = '--'
        print(f'{fid:10s}  {gt_k:>4d}  {str(pe):>6s}  {str(ps):>6s}  {se:>4d}  {ss:>4d}  {change}')

    print(f"\n总结:")
    print(f"  Energy VAD: {correct_e}/{n} = {correct_e/n*100:.1f}%")
    print(f"  Silero VAD: {correct_s}/{n} = {correct_s/n*100:.1f}%")
    better = sum(1 for fid in TEST_25 if fid in gt and preds_e.get(fid) != gt[fid] and preds_s.get(fid) == gt[fid])
    worse  = sum(1 for fid in TEST_25 if fid in gt and preds_e.get(fid) == gt[fid] and preds_s.get(fid) != gt[fid])
    print(f"  Silero 修正: {better} 个, 引入新错误: {worse} 个")
