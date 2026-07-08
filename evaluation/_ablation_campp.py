"""CAM++ 参数网格搜索 — 25文件"""
import json, os, sys, time
sys.path.insert(0, '.')
from rttm_parser import parse_rttm

# ---- 导入在线追踪模块 ----
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from blind_count import (
    TEST_25, AUDIO, SR_TARGET, get_embedding_model,
    energy_vad, merge_short_segments, extract_timeline_embeddings,
    online_track
)

CACHE_PATH = '_campp_timelines_25.json'
EVAL_DIR = os.path.dirname(os.path.abspath(__file__))
RTTM_DIR = r'c:\Users\hp\Desktop\SoundCUDA-main\labels\dev'

# ---- 第一步: 加载/提取所有文件的 timeline embeddings ----
def load_or_extract_timelines():
    if os.path.exists(CACHE_PATH):
        print(f"[Cache] 加载缓存 {CACHE_PATH}")
        with open(CACHE_PATH) as f:
            raw = json.load(f)
        # 转换回 numpy
        timelines = {}
        for fid, entries in raw.items():
            timelines[fid] = [(e[0], np.array(e[1])) for e in entries]
        return timelines

    print("[Extract] 提取所有文件嵌入 (CAM++)...")
    model = get_embedding_model()
    import soundfile as sf
    from scipy.signal import resample as scipy_resample

    timelines = {}
    wav_files = [(f, os.path.join(AUDIO, f + '.wav')) for f in TEST_25
                 if os.path.exists(os.path.join(AUDIO, f + '.wav'))]

    for i, (fid, wav) in enumerate(wav_files):
        t0 = time.time()
        y, sr_file = sf.read(wav, dtype='float32')
        if y.ndim > 1:
            y = y.mean(axis=1)
        if sr_file != SR_TARGET:
            y = scipy_resample(y, int(len(y) * SR_TARGET / sr_file))

        raw_segs = energy_vad(y, SR_TARGET)
        voice_segs = merge_short_segments(raw_segs)
        if len(voice_segs) == 0:
            timelines[fid] = []
        else:
            tl = extract_timeline_embeddings(y, SR_TARGET, voice_segs, model)
            timelines[fid] = [(float(t), emb.tolist()) for t, emb in tl]
        print(f"  [{i+1:>2}/{len(wav_files)}] {fid}  segs={len(timelines[fid])}  {time.time()-t0:.0f}s")

    # 保存缓存 (entries 已经是 (float, list) 格式)
    save = {fid: [(float(t), emb) for t, emb in entries]
            for fid, entries in timelines.items()}
    with open(CACHE_PATH, 'w') as f:
        json.dump(save, f)
    print(f"[Cache] 已保存 {CACHE_PATH}")
    return timelines


# ---- 第二步: 加载真实值 ----
def load_ground_truth():
    gt = {}
    for fid in TEST_25:
        rttm_path = os.path.join(RTTM_DIR, fid + '.rttm')
        if os.path.exists(rttm_path):
            segs = parse_rttm(rttm_path)
            gt[fid] = len(set(s[2] for s in segs))
    return gt


# ---- 第三步: 网格搜索 ----
def grid_search(timelines, gt):
    thresholds = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55]
    emas = [0.25, 0.35, 0.45]
    min_confirms = [2, 3, 4]

    print(f"\n{'='*80}")
    print(f"参数网格: {len(thresholds)} thresholds x {len(emas)} EMAs x {len(min_confirms)} MIN_CONFIRMs = {len(thresholds)*len(emas)*len(min_confirms)} 组")
    print(f"{'='*80}")

    results = []
    n_files = len([f for f in TEST_25 if f in gt and f in timelines and len(timelines[f]) > 0])

    for th in thresholds:
        for em in emas:
            for mc in min_confirms:
                correct = 0
                abs_err_sum = 0
                details = {}

                for fid in TEST_25:
                    if fid not in gt or fid not in timelines:
                        continue
                    tl = timelines[fid]
                    if len(tl) < 2:
                        pred = 1
                    else:
                        pred, _ = online_track(tl, threshold=th, ema=em, min_confirm=mc)
                    true_k = gt[fid]
                    details[fid] = pred
                    if pred == true_k:
                        correct += 1
                    abs_err_sum += abs(pred - true_k)

                acc = correct / n_files * 100
                mae = abs_err_sum / n_files
                results.append((acc, mae, th, em, mc, correct))
                print(f"  th={th:.2f}  ema={em:.2f}  mc={mc}  →  exact={correct}/{n_files}={acc:.1f}%  MAE={mae:.3f}")

    # 排序输出最佳
    results.sort(key=lambda x: (-x[0], x[1]))  # 先按准确率降序，再按MAE升序
    print(f"\n{'='*80}")
    print("TOP 10:")
    print(f"{'Rank':5s}  {'Acc':>6s}  {'MAE':>6s}  {'th':>6s}  {'ema':>6s}  {'mc':>4s}")
    print('-'*45)
    for rank, (acc, mae, th, em, mc, corr) in enumerate(results[:10]):
        print(f"  {rank+1:>2}    {acc:>5.1f}%  {mae:>5.2f}  {th:>5.2f}  {em:>5.2f}  {mc:>3}")

    # 当前配置的表现
    print(f"\n当前配置 (th=0.40 ema=0.35 mc=3):")
    for r in results:
        if abs(r[2] - 0.40) < 0.001 and abs(r[3] - 0.35) < 0.001 and r[4] == 3:
            print(f"  exact={r[5]}/{n_files}={r[0]:.1f}%  MAE={r[1]:.3f}")

    return results


if __name__ == '__main__':
    os.chdir(EVAL_DIR)
    gt = load_ground_truth()
    timelines = load_or_extract_timelines()
    grid_search(timelines, gt)
