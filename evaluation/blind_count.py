#!/usr/bin/env python3
"""
纯盲说话人计数 v3 — 在线追踪法 (Online Tracking)
模拟人耳机制：按时间顺序逐段判断 "新人物 or 已知人物"
全程不碰 RTTM 标签，不预设 k
"""
import os, sys, time, warnings
warnings.filterwarnings('ignore')

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE = os.path.dirname(PROJECT)
sys.path.insert(0, PROJECT)

import numpy as np
import torch
import json
import shutil

# torch 2.6 兼容: 强制 weights_only=False
_t_load = torch.load
torch.load = lambda *a, **kw: _t_load(*a, **{**kw, 'weights_only': False})

# Windows 符号链接修复: 强制用 copy 代替 symlink
import pathlib
if hasattr(pathlib.WindowsPath, 'symlink_to'):
    _orig_symlink = pathlib.WindowsPath.symlink_to
    def _copy_instead_of_symlink(self, target, target_is_directory=False):
        target_path = pathlib.Path(target)
        if target_path.is_file():
            shutil.copy2(str(target_path), str(self))
        elif target_path.is_dir():
            shutil.copytree(str(target_path), str(self), dirs_exist_ok=True)
    pathlib.WindowsPath.symlink_to = _copy_instead_of_symlink

AUDIO = os.path.join(BASE, 'voxconverse_dev_wav', 'audio')
SR_TARGET = 16000
VAD_FRAME_MS = 25
VAD_HOP_MS = 10

# ---- 在线追踪参数 ----
THRESHOLD_MATCH = 0.55      # 余弦相似度 > 此值 → 同一人 (CAM++ 512维, 网格搜索最优)
EMA_ALPHA = 0.25            # 声纹档案更新权重 (越小越保守)
MIN_CONFIRM = 4             # 候选说话人至少被确认 N 次才成为正式档案
MIN_SEG_DUR = 3.5           # 最短有效语音段 (秒)，过短则合并或丢弃
CHUNK_DUR = 3.0             # 每段提取嵌入的窗口长度 (秒)
DUAL_PASS = False           # True=双向追踪 False=仅正向

# ---- CAM++ VoxCeleb 512维嵌入 (FunASR 架构 + ModelScope 权重) ----
_embedding_model = None
_FBANK_DIM = 80
_EMB_SIZE = 512
_CKPT_PATH = r'C:\Users\hp\.cache\modelscope\models\damo--speech_campplus_sv_en_voxceleb_16k\snapshots\master\campplus_voxceleb.bin'

def get_embedding_model():
    global _embedding_model
    if _embedding_model is None:
        from funasr.models.campplus.model import CAMPPlus
        print("[Embedding] 加载 VoxCeleb CAM++ (512维)...")
        _embedding_model = CAMPPlus(embedding_size=_EMB_SIZE)
        sd = torch.load(_CKPT_PATH, map_location='cpu')
        _embedding_model.load_state_dict(sd, strict=True)
        _embedding_model.eval()
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        _embedding_model = _embedding_model.to(device)
        print(f"[Embedding] 加载完成 (device={device})")
    return _embedding_model


def extract_one_embedding(chunk, sr, model):
    """从一段音频提取 CAM++ 嵌入向量 (512-dim)"""
    from funasr.models.campplus.utils import extract_feature
    from funasr.utils.load_utils import load_audio_text_image_video
    device = next(model.parameters()).device
    audio_list = load_audio_text_image_video([chunk], fs=16000, audio_fs=sr, data_type='sound')
    feats, _, _ = extract_feature(audio_list)
    feats = feats.to(device=device).to(torch.float32)
    with torch.no_grad():
        emb = model(feats)
    return emb.cpu().numpy().squeeze()


def energy_vad(y, sr, percentile=15):
    """纯信号处理 VAD"""
    frame_len = int(sr * VAD_FRAME_MS / 1000)
    hop_len = int(sr * VAD_HOP_MS / 1000)
    n_frames = (len(y) - frame_len) // hop_len + 1

    energies = np.array([
        np.sum(y[i*hop_len : i*hop_len+frame_len]**2)
        for i in range(n_frames)
    ])
    if len(energies) == 0:
        return []

    threshold = np.percentile(energies, percentile)
    is_voice = energies > threshold

    kernel = 5
    is_voice = np.convolve(is_voice.astype(int), np.ones(kernel), mode='same') > (kernel // 2)

    segments = []
    in_seg = False
    seg_start = 0
    min_gap_frames = int(0.3 * 1000 / VAD_HOP_MS)

    for i in range(len(is_voice)):
        t = i * VAD_HOP_MS / 1000
        if is_voice[i] and not in_seg:
            in_seg = True
            seg_start = t
        elif not is_voice[i] and in_seg:
            gap_end = min(i + min_gap_frames, len(is_voice))
            if np.any(is_voice[i:gap_end]):
                continue
            seg_end = t
            if seg_end - seg_start >= 0.5:
                segments.append((seg_start, seg_end))
            in_seg = False
    if in_seg:
        t_end = len(is_voice) * VAD_HOP_MS / 1000
        if t_end - seg_start >= 0.5:
            segments.append((seg_start, t_end))

    return segments


def merge_short_segments(voice_segs, max_gap=1.0):
    """合并相邻短段：间隔 < max_gap 且合并后不超长则合并，丢弃过短孤立段"""
    if len(voice_segs) == 0:
        return []
    merged = []
    cur_start, cur_end = voice_segs[0]
    for s, e in voice_segs[1:]:
        if s - cur_end < max_gap:
            cur_end = e
        else:
            if cur_end - cur_start >= MIN_SEG_DUR:
                merged.append((cur_start, cur_end))
            cur_start, cur_end = s, e
    if cur_end - cur_start >= MIN_SEG_DUR:
        merged.append((cur_start, cur_end))
    return merged


def extract_timeline_embeddings(y, sr, voice_segs, model):
    """
    按时间顺序从每个语音段提取嵌入
    段长 2.5-8s: 取段中心 CHUNK_DUR=3s 窗口
    段长 >8s:    滑动 3s 窗口切多个嵌入
    返回: [(time_mid, embedding), ...]
    """
    results = []

    for seg_start, seg_end in voice_segs:
        seg_dur = seg_end - seg_start

        if seg_dur <= 8.0:
            # 取段中心附近 3s 窗口
            center = (seg_start + seg_end) / 2
            half = CHUNK_DUR / 2
            chunk_start = max(seg_start, center - half)
            chunk_end = min(seg_end, center + half)
            s = int(chunk_start * sr)
            e = int(chunk_end * sr)
            chunk = y[s:e]
            emb = extract_one_embedding(chunk, sr, model)
            results.append((center, emb))

        else:
            # 长段: 滑动窗口切多个 3s 子段
            step_s = 2.0
            t = seg_start
            while t + CHUNK_DUR <= seg_end:
                s = int(t * sr)
                e = int((t + CHUNK_DUR) * sr)
                chunk = y[s:e]
                emb = extract_one_embedding(chunk, sr, model)
                results.append((t + CHUNK_DUR / 2, emb))
                t += step_s

    return results


# ===== 核心：在线追踪算法 =====

def online_track(timeline_embeddings, threshold=None, ema=None, min_confirm=None):
    """
    在线追踪法 — 模拟人耳逐段聆听机制
    
    参数可覆盖模块级默认值 (用于 ablation 实验)
    """
    if threshold is None: threshold = THRESHOLD_MATCH
    if ema is None: ema = EMA_ALPHA
    if min_confirm is None: min_confirm = MIN_CONFIRM

    profiles = []       # [(embedding, count), ...] 已确认的说话人档案
    candidates = []     # [(embedding, count), ...] 候选说话人 (待确认)
    trace = []          # 调试日志

    for seg_idx, (t_mid, emb_raw) in enumerate(timeline_embeddings):
        emb = emb_raw / (np.linalg.norm(emb_raw) + 1e-8)

        # ---- 第一步: 和已确认档案比对 ----
        matched = False
        if profiles:
            sims = [np.dot(emb, p[0]) for p in profiles]
            max_sim = max(sims)
            best_idx = int(np.argmax(sims))

            if max_sim > threshold:
                old_emb, cnt = profiles[best_idx]
                new_emb = ema * emb + (1 - ema) * old_emb
                new_emb = new_emb / (np.linalg.norm(new_emb) + 1e-8)
                profiles[best_idx] = (new_emb, cnt + 1)
                trace.append(f"  seg{seg_idx} t={t_mid:.1f}s → 档案{best_idx} (sim={max_sim:.3f})")
                matched = True

        if matched:
            continue

        # ---- 第二步: 和候选档案比对 ----
        matched_candidate = False
        if candidates:
            sims_c = [np.dot(emb, c[0]) for c in candidates]
            max_sim_c = max(sims_c)
            best_ci = int(np.argmax(sims_c))

            if max_sim_c > threshold:
                old_emb, cnt = candidates[best_ci]
                new_emb = ema * emb + (1 - ema) * old_emb
                new_emb = new_emb / (np.linalg.norm(new_emb) + 1e-8)
                new_cnt = cnt + 1
                trace.append(f"  seg{seg_idx} t={t_mid:.1f}s → 候选{best_ci} "
                             f"(sim={max_sim_c:.3f} cnt→{new_cnt})")

                if new_cnt >= min_confirm:
                    profiles.append((new_emb, new_cnt))
                    candidates.pop(best_ci)
                    trace.append(f"    ↑ 候选升级为档案{len(profiles)-1}")
                else:
                    candidates[best_ci] = (new_emb, new_cnt)
                matched_candidate = True

        if matched_candidate:
            continue

        # ---- 第三步: 全新声音 → 创建候选 ----
        candidates.append((emb, 1))
        trace.append(f"  seg{seg_idx} t={t_mid:.1f}s → 新候选{len(candidates)-1}")

    # ---- 统计人数 ----
    n_confirmed = len(profiles)

    # 统计"几乎确认"的候选
    n_strong_candidates = sum(1 for _, cnt in candidates if cnt >= max(1, min_confirm - 1))

    if n_confirmed == 0:
        return max(1, len(candidates)), trace

    result = n_confirmed + n_strong_candidates
    return max(1, result), trace


def online_track_dual_pass(timeline_embeddings, threshold=None, ema=None, min_confirm=None):
    """
    双向在线追踪 — 正向+反向各扫一遍，合并去重
    解决正向扫描时后出现但声纹相似的人被 EMA 吞掉的问题
    """
    if threshold is None: threshold = THRESHOLD_MATCH
    if ema is None: ema = EMA_ALPHA
    if min_confirm is None: min_confirm = MIN_CONFIRM

    if len(timeline_embeddings) < 2:
        return 1, []

    # 正向扫描
    def run_pass(timeline):
        """返回: (确认档案列表[(emb, cnt)], 候选列表[(emb, cnt)], trace)"""
        profiles = []
        candidates = []
        trace = []
        for seg_idx, (t_mid, emb_raw) in enumerate(timeline):
            emb = emb_raw / (np.linalg.norm(emb_raw) + 1e-8)
            matched = False
            if profiles:
                sims = [np.dot(emb, p[0]) for p in profiles]
                max_sim = max(sims)
                if max_sim > threshold:
                    old_emb, cnt = profiles[int(np.argmax(sims))]
                    new_emb = ema * emb + (1 - ema) * old_emb
                    new_emb = new_emb / (np.linalg.norm(new_emb) + 1e-8)
                    profiles[int(np.argmax(sims))] = (new_emb, cnt + 1)
                    matched = True
            if matched:
                continue
            matched_c = False
            if candidates:
                sims_c = [np.dot(emb, c[0]) for c in candidates]
                max_sim_c = max(sims_c)
                if max_sim_c > threshold:
                    old_emb, cnt = candidates[int(np.argmax(sims_c))]
                    new_emb = ema * emb + (1 - ema) * old_emb
                    new_emb = new_emb / (np.linalg.norm(new_emb) + 1e-8)
                    new_cnt = cnt + 1
                    if new_cnt >= min_confirm:
                        profiles.append((new_emb, new_cnt))
                        candidates.pop(int(np.argmax(sims_c)))
                    else:
                        candidates[int(np.argmax(sims_c))] = (new_emb, new_cnt)
                    matched_c = True
            if matched_c:
                continue
            candidates.append((emb, 1))
        return profiles, candidates, trace

    profiles_fwd, cands_fwd, trace_fwd = run_pass(timeline_embeddings)
    profiles_rev, cands_rev, trace_rev = run_pass(list(reversed(timeline_embeddings)))

    all_trace = trace_fwd + ["--- 反向扫描 ---"] + trace_rev

    # 合并: 反向档案中与正向不重复的，加入最终结果
    merged_profiles = list(profiles_fwd)
    merge_threshold = threshold + 0.05  # 去重阈值稍高，避免把同一个人当两个

    for rev_emb, rev_cnt in profiles_rev:
        is_new = True
        for fwd_emb, _ in merged_profiles:
            sim = np.dot(rev_emb, fwd_emb)
            if sim > merge_threshold:
                is_new = False
                break
        if is_new:
            merged_profiles.append((rev_emb, rev_cnt))

    # 候选也类似处理
    all_cand_embs = [(e, c) for e, c in cands_fwd]
    for rev_emb, rev_cnt in cands_rev:
        # 检查是否与已确认档案重复
        dup = False
        for fwd_emb, _ in merged_profiles:
            if np.dot(rev_emb, fwd_emb) > merge_threshold:
                dup = True; break
        if not dup:
            for ce, _ in all_cand_embs:
                if np.dot(rev_emb, ce) > merge_threshold:
                    dup = True; break
        if not dup:
            all_cand_embs.append((rev_emb, rev_cnt))

    n_confirmed = len(merged_profiles)
    n_strong = sum(1 for _, cnt in all_cand_embs if cnt >= max(1, min_confirm - 1))

    if n_confirmed == 0:
        return max(1, len(all_cand_embs)), all_trace

    return max(1, n_confirmed + n_strong), all_trace


# ===== 主入口 =====

def count_speakers(audio_path, model, verbose=False, **track_kwargs):
    """纯盲推断说话人数 — 在线追踪法"""
    import soundfile as sf
    from scipy.signal import resample as scipy_resample

    y, sr_file = sf.read(audio_path, dtype='float32')
    if y.ndim > 1:
        y = y.mean(axis=1)

    if sr_file != SR_TARGET:
        y = scipy_resample(y, int(len(y) * SR_TARGET / sr_file))

    sr = SR_TARGET
    dur = len(y) / sr

    # VAD + 合并短段
    raw_segs = energy_vad(y, sr)
    voice_segs = merge_short_segments(raw_segs)
    if len(voice_segs) == 0:
        return 1, dur, 0, []

    # 按时间顺序提取嵌入
    timeline = extract_timeline_embeddings(y, sr, voice_segs, model)
    n_segs = len(timeline)
    if n_segs < 2:
        return 1, dur, n_segs, []

    # 在线追踪 (正向 / 双向)
    if DUAL_PASS:
        k, trace = online_track_dual_pass(timeline, **track_kwargs)
    else:
        k, trace = online_track(timeline, **track_kwargs)

    if verbose:
        for line in trace:
            print(line)

    return k, dur, n_segs, trace


# ---- 25文件快速测试集 ----
TEST_25 = ['bkwns','abjxc','syiwe','qppll','cobal','oenox','bwzyf','jiqvr','jyirt',
           'hiyis','plbbw','vysqj','sikkm','wjhgf','lknjp','mevkw','kctgl','zfkap',
           'iqtde','xiglo','jsmbi','qydmg','akthc','exymw','kbkon']
MODE_25_ONLY = False  # True=只测25条  False=全216条


def main():
    import glob
    if MODE_25_ONLY:
        wav_files = [os.path.join(AUDIO, f + '.wav') for f in TEST_25
                     if os.path.exists(os.path.join(AUDIO, f + '.wav'))]
        out_name = 'blind_online_25.json'
    else:
        wav_files = sorted(glob.glob(os.path.join(AUDIO, '*.wav')))
        out_name = 'blind_online_track.json'
    total = len(wav_files)

    mode_str = "双向追踪" if DUAL_PASS else "正向追踪"
    print("=" * 60)
    print(f"  纯盲说话人计数 — 在线追踪法 {mode_str} ({total}条)")
    print(f"  阈值: match>{THRESHOLD_MATCH:.2f}  min确认={MIN_CONFIRM}  EMA={EMA_ALPHA}")
    print("=" * 60)

    model = get_embedding_model()

    predictions = {}
    out_path = os.path.join(PROJECT, 'evaluation', out_name)

    for i, wav in enumerate(wav_files):
        fid = os.path.splitext(os.path.basename(wav))[0]

        t0 = time.time()
        k, dur, n_segs, trace = count_speakers(wav, model, verbose=False)
        elapsed = time.time() - t0

        n_profiles = len([l for l in trace if '档案' in l and '↑' not in l])
        n_cands = len([l for l in trace if '新候选' in l])

        print(f"  [{i+1:>3}/{total}] {fid}  dur={dur:5.0f}s  "
              f"预测={k:>2}人  segs={n_segs:>3}  档案={n_profiles}  候选={n_cands}  {elapsed:4.0f}s")
        predictions[fid] = k

        if (i + 1) % 10 == 0:
            with open(out_path, 'w') as f:
                json.dump(predictions, f, indent=2, ensure_ascii=False)

    with open(out_path, 'w') as f:
        json.dump(predictions, f, indent=2, ensure_ascii=False)
    print(f"\n预测结果已保存到 evaluation/{out_name}")


if __name__ == '__main__':
    main()
