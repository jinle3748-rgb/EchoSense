#!/usr/bin/env python3
"""
纯盲说话人计数 v2 — 两阶段架构
Tier1 筛选器 → Tier2 谱聚类(轻量) / Tier3 DBSCAN密度计数(重量)
全程不碰 RTTM 标签
"""
import os, sys, time, warnings
warnings.filterwarnings('ignore')

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE = os.path.dirname(PROJECT)
sys.path.insert(0, PROJECT)

import numpy as np
import torch
import json

AUDIO = os.path.join(BASE, 'voxconverse_dev_wav', 'audio')
SR_TARGET = 16000
VAD_FRAME_MS = 25
VAD_HOP_MS = 10

# ---- ECAPA ----
os.environ.setdefault('HF_ENDPOINT', 'https://hf-mirror.com')

_classifier = None
def get_classifier():
    global _classifier
    if _classifier is None:
        from speechbrain.inference.speaker import EncoderClassifier
        print("[ECAPA] 加载模型...")
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        _classifier = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            run_opts={"device": device}
        )
        print("[ECAPA] 加载完成")
    return _classifier


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


def extract_embeddings_dense(y, sr, classifier, chunk_dur=1.5, overlap=0.5):
    """密集采样：更短段长、更大重叠 → 更多段，用于多人场景"""
    step = chunk_dur * (1 - overlap)
    chunk_samps = int(chunk_dur * sr)
    step_samps = int(step * sr)

    embeddings = []
    for start in range(0, max(1, len(y) - chunk_samps + 1), max(1, step_samps)):
        end = min(start + chunk_samps, len(y))
        chunk = y[start:end]
        if len(chunk) < sr * 0.5:
            continue
        target_len = int(np.ceil(len(chunk) / sr) * sr)
        chunk = np.pad(chunk, (0, max(0, target_len - len(chunk))))
        with torch.no_grad():
            emb = classifier.encode_batch(
                torch.from_numpy(chunk).unsqueeze(0)
            ).squeeze().cpu().numpy()
        embeddings.append(emb)

    if len(embeddings) == 0:
        return np.zeros((0, 192))
    return np.stack(embeddings)


def extract_embeddings_sparse(y, sr, classifier, chunk_dur=3.0, overlap=0.25):
    """稀疏采样：更长的段、更少重叠 → 嵌入更稳定，用于少人场景"""
    step = chunk_dur * (1 - overlap)
    chunk_samps = int(chunk_dur * sr)
    step_samps = int(step * sr)

    embeddings = []
    for start in range(0, max(1, len(y) - chunk_samps + 1), max(1, step_samps)):
        end = min(start + chunk_samps, len(y))
        chunk = y[start:end]
        if len(chunk) < sr * 0.5:
            continue
        target_len = int(np.ceil(len(chunk) / sr) * sr)
        chunk = np.pad(chunk, (0, max(0, target_len - len(chunk))))
        with torch.no_grad():
            emb = classifier.encode_batch(
                torch.from_numpy(chunk).unsqueeze(0)
            ).squeeze().cpu().numpy()
        embeddings.append(emb)

    if len(embeddings) == 0:
        return np.zeros((0, 192))
    return np.stack(embeddings)


# ===== Tier 1: 筛选器 =====

def screen_light_or_heavy(embeddings, dur):
    """
    快速判断是轻量级(≤4人)还是重量级(5+人)
    使用嵌入多样性(平均两两距离) + 段数 + 时长 综合判断
    返回: 'light' 或 'heavy'
    """
    N = len(embeddings)
    if N < 5:
        return 'light'

    # 归一化
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    X = embeddings / norms

    # 嵌入多样性: 随机抽 min(N, 50) 个段计算平均成对余弦距离
    n_sample = min(N, 50)
    idx = np.random.choice(N, n_sample, replace=False)
    X_sample = X[idx]
    S_sample = X_sample @ X_sample.T
    # 平均成对距离 = 1 - 平均相似度 (去掉对角线)
    mean_sim = (np.sum(S_sample) - n_sample) / (n_sample * (n_sample - 1))
    diversity = 1 - mean_sim

    # 段密度: 每秒平均语音段数
    seg_density = N / max(dur, 1)

    # 综合评分
    score = diversity * 5 + seg_density * 2 + min(dur / 300, 1)

    if score > 2.5:
        return 'heavy'
    return 'light'


# ===== Tier 2: 谱聚类 (轻量) =====

def count_spectral(embeddings):
    """谱聚类 + 轮廓系数，用于 1-4 人精确计数"""
    from sklearn.cluster import SpectralClustering
    from sklearn.metrics import silhouette_score

    N = len(embeddings)
    if N < 3:
        return 1

    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    X = embeddings / norms
    S = X @ X.T

    mean_sim = (np.sum(S) - N) / (N * (N - 1))
    if mean_sim > 0.80:
        return 1

    max_k = min(10, N // 3, N - 1)
    if max_k < 2:
        return 1

    scores = []
    for k in range(2, max_k + 1):
        try:
            sc = SpectralClustering(n_clusters=k, affinity='precomputed',
                                    random_state=42, assign_labels='kmeans',
                                    n_init=10)
            labels = sc.fit_predict(np.clip(S, 0, 1))
            if len(set(labels)) < 2:
                continue
            score = silhouette_score(X, labels)
            scores.append((k, score))
        except Exception:
            continue

    if not scores:
        return 1

    best_k, best_score = max(scores, key=lambda x: x[1])
    if best_score < 0.08:
        return 1
    return best_k


# ===== Tier 3: DBSCAN 密度计数 (重量) =====

def count_dbscan(embeddings):
    """
    DBSCAN 密度峰值计数 — 不预设 k，让数据自己形成簇
    自动选择 eps 参数 (基于 k-距离图拐点)
    返回: 预测说话人数
    """
    from sklearn.neighbors import NearestNeighbors
    from sklearn.cluster import DBSCAN

    N = len(embeddings)
    if N < 5:
        return 1

    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    X = embeddings / norms

    # 余弦距离 = 1 - 余弦相似度
    # 计算每个点到第 k 近邻的距离，用于自动选择 eps
    k_neighbors = min(10, N // 5)
    if k_neighbors < 2:
        return 1

    nbrs = NearestNeighbors(n_neighbors=k_neighbors, metric='cosine')
    nbrs.fit(X)
    distances, _ = nbrs.kneighbors(X)
    k_dist = np.sort(distances[:, -1])  # 第k近邻距离，排序

    # 找拐点：k-距离曲线最大曲率处
    if len(k_dist) < 5:
        return 1

    # 二阶差分找曲率最大点
    d2 = np.diff(k_dist, 2)
    if len(d2) == 0:
        eps = np.median(k_dist)
    else:
        knee = np.argmax(np.abs(d2)) + 1
        eps = k_dist[min(knee, len(k_dist) - 1)]

    # 防止 eps 过小或过大
    eps = max(0.05, min(eps, 0.5))

    # DBSCAN 聚类
    min_samples = max(3, N // 30)
    db = DBSCAN(eps=eps, min_samples=min_samples, metric='cosine')
    labels = db.fit_predict(X)

    # 统计有效簇数 (排除噪声标签 -1)
    unique_clusters = set(labels) - {-1}
    n_clusters = len(unique_clusters)

    if n_clusters == 0:
        return 1

    return n_clusters


# ===== 主入口 =====

def count_speakers(audio_path, classifier):
    """纯盲推断说话人数"""
    import soundfile as sf
    from scipy.signal import resample as scipy_resample

    y, sr_file = sf.read(audio_path, dtype='float32')
    if y.ndim > 1:
        y = y.mean(axis=1)

    if sr_file != SR_TARGET:
        y = scipy_resample(y, int(len(y) * SR_TARGET / sr_file))

    sr = SR_TARGET
    dur = len(y) / sr

    # VAD
    voice_segs = energy_vad(y, sr)
    if len(voice_segs) == 0:
        return 1, dur, 0, 'none'

    voice_signal = np.concatenate([
        y[int(s*sr):int(e*sr)] for s, e in voice_segs
    ])

    # Tier 1: 快速筛选 (稀疏采样 ~30段)
    emb_quick = extract_embeddings_sparse(voice_signal, sr, classifier,
                                          chunk_dur=3.0, overlap=0.5)
    n_segs = len(emb_quick)
    if n_segs < 3:
        return 1, dur, n_segs, 'tiny'

    tier = screen_light_or_heavy(emb_quick, dur)

    k = count_spectral(emb_quick)
    return k, dur, n_segs, 'spectral'


def main():
    import glob
    wav_files = sorted(glob.glob(os.path.join(AUDIO, '*.wav')))
    total = len(wav_files)

    print("=" * 60)
    print(f"  纯盲说话人计数 — 谱聚类+轮廓系数 (全{total}条)")
    print("=" * 60)

    classifier = get_classifier()

    predictions = {}
    out_path = os.path.join(PROJECT, 'evaluation', 'blind_216.json')
    for i, wav in enumerate(wav_files):
        fid = os.path.splitext(os.path.basename(wav))[0]

        t0 = time.time()
        k, dur, n_segs, method = count_speakers(wav, classifier)
        elapsed = time.time() - t0

        print(f"  [{i+1:>3}/{total}] {fid}  dur={dur:5.0f}s  "
              f"预测={k:>2}人  segs={n_segs:>3}  {elapsed:4.0f}s")
        predictions[fid] = k

        # 每10条保存一次
        if (i + 1) % 10 == 0:
            with open(out_path, 'w') as f:
                json.dump(predictions, f, indent=2, ensure_ascii=False)

    with open(out_path, 'w') as f:
        json.dump(predictions, f, indent=2, ensure_ascii=False)
    print(f"\n预测结果已保存到 evaluation/blind_216.json")


if __name__ == '__main__':
    main()
