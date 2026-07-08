"""诊断 kbkon / bwzyf 漏人原因"""
import json, os, sys, time
import numpy as np
sys.path.insert(0, '.')
from rttm_parser import parse_rttm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from blind_count import (
    AUDIO, SR_TARGET, get_embedding_model,
    energy_vad, merge_short_segments, extract_timeline_embeddings,
    online_track
)

RTTM_DIR = r'c:\Users\hp\Desktop\SoundCUDA-main\labels\dev'
CASES = ['kbkon', 'bwzyf']

for fid in CASES:
    print(f"\n{'='*70}")
    print(f"  诊断: {fid}")
    print(f"{'='*70}")

    # ---- Ground Truth ----
    rttm_path = os.path.join(RTTM_DIR, fid + '.rttm')
    segs = parse_rttm(rttm_path)
    spk_set = sorted(set(s[2] for s in segs))
    print(f"真实说话人: {len(spk_set)} 人 → {spk_set}")
    print(f"RTTM段数: {len(segs)}")

    # 每个真实说话人的时间覆盖
    for spk in spk_set:
        spk_segs = [(s, e) for s, e, sp in segs if sp == spk]
        total_spk = sum(e - s for s, e in spk_segs)
        print(f"  {spk}: {len(spk_segs)}段, 共{total_spk:.0f}s")

    # ---- 提取嵌入 + 在线追踪 ----
    import soundfile as sf
    from scipy.signal import resample as scipy_resample

    wav = os.path.join(AUDIO, fid + '.wav')
    y, sr_file = sf.read(wav, dtype='float32')
    if y.ndim > 1:
        y = y.mean(axis=1)
    if sr_file != SR_TARGET:
        y = scipy_resample(y, int(len(y) * SR_TARGET / sr_file))

    model = get_embedding_model()
    raw_segs = energy_vad(y, SR_TARGET)
    voice_segs = merge_short_segments(raw_segs)
    timeline = extract_timeline_embeddings(y, SR_TARGET, voice_segs, model)
    print(f"\n有效语音段: {len(voice_segs)}, 提取嵌入: {len(timeline)}")

    # ---- 在线追踪 (verbose) ----
    print(f"\n--- 追踪过程 (th=0.55 ema=0.25 mc=4) ---")
    k, trace = online_track(timeline, threshold=0.55, ema=0.25, min_confirm=4)
    for line in trace:
        print(line)

    # ---- 嵌入相似度分析 ----
    print(f"\n--- 嵌入间最大/最小余弦相似度 ---")
    embs = [e / (np.linalg.norm(e) + 1e-8) for _, e in timeline]
    n = len(embs)
    sim_matrix = np.zeros((n, n))
    pair_max = -1; pair_max_ij = (0, 0)
    pair_min = 2;  pair_min_ij = (0, 0)
    for i in range(n):
        for j in range(i+1, n):
            s = np.dot(embs[i], embs[j])
            sim_matrix[i, j] = s
            if s > pair_max:
                pair_max = s; pair_max_ij = (i, j)
            if s < pair_min:
                pair_min = s; pair_min_ij = (i, j)

    # 统计相似度分布
    all_sims = [sim_matrix[i, j] for i in range(n) for j in range(i+1, n)]
    all_sims = np.array(all_sims)
    print(f"  段间相似度: min={pair_min:.3f}  max={pair_max:.3f}  mean={all_sims.mean():.3f}  std={all_sims.std():.3f}")
    print(f"  最高相似段: seg{pair_max_ij[0]} (t={timeline[pair_max_ij[0]][0]:.1f}s) ↔ seg{pair_max_ij[1]} (t={timeline[pair_max_ij[1]][0]:.1f}s)")
    print(f"  最低相似段: seg{pair_min_ij[0]} (t={timeline[pair_min_ij[0]][0]:.1f}s) ↔ seg{pair_min_ij[1]} (t={timeline[pair_min_ij[1]][0]:.1f}s)")

    # 直方图
    bins = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    hist, _ = np.histogram(all_sims, bins=bins)
    print(f"  分布: ", end="")
    for i in range(len(bins)-1):
        bar = '█' * max(1, int(hist[i] / max(1, hist.max()) * 30))
        print(f"[{bins[i]:.1f}-{bins[i+1]:.1f}]:{hist[i]:3d} {bar}")
