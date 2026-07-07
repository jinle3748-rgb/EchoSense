"""
ECAPA-TDNN 说话人分析模块
- 使用 speechbrain 预训练 ECAPA-TDNN (VoxCeleb) 提取192维声纹嵌入
- 滑动窗口分段 + 轮廓系数聚类确定说话人数
- 静音检测(RMS) + F0性别估计 + 后验合并
- 独立运行: python ecapa_speaker.py <音频文件>
- GUI调用: from ECAPA-TDNN.ecapa_speaker import analyze_speakers
"""

import os, sys, warnings
warnings.filterwarnings('ignore')

# 使用国内 HuggingFace 镜像 (解决连接超时问题)
os.environ.setdefault('HF_ENDPOINT', 'https://hf-mirror.com')

import numpy as np
from scipy.io import wavfile
from scipy.signal import resample, butter, filtfilt, correlate
import torch

# ---- 模型缓存 ----
_classifier = None
_device = None

def _get_device():
    global _device
    if _device is None:
        _device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return _device

def _get_classifier():
    global _classifier
    if _classifier is None:
        from speechbrain.inference.speaker import EncoderClassifier
        device = _get_device()
        device_str = "cuda:0" if device.type == "cuda" else "cpu"
        print(f"[ECAPA] 加载预训练模型 speechbrain/spkrec-ecapa-voxceleb 到 {device_str}...")
        _classifier = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            run_opts={"device": device_str}
        )
        print("[ECAPA] 模型加载完成")
    return _classifier


def extract_pitch(y, sr=16000, min_f0=80, max_f0=400):
    """
    提取基频 F0 (音高)，用于辅助性别判断
    男性: 80-160Hz, 女性: 160-400Hz
    """
    nyq = sr / 2
    b, a = butter(4, [min_f0/nyq, min(max_f0/nyq, 0.95)], btype='band')
    y_filt = filtfilt(b, a, y)

    n = len(y_filt)
    if n < sr // min_f0:
        return np.array([0.0])

    autocorr = correlate(y_filt, y_filt, mode='full')
    autocorr = autocorr[n-1:]

    min_lag = sr // max_f0
    max_lag = min(sr // min_f0, n-1)
    if max_lag <= min_lag:
        return np.array([0.0])

    search = autocorr[min_lag:max_lag]
    if len(search) == 0:
        return np.array([0.0])

    peak_lag = np.argmax(search) + min_lag
    if peak_lag == 0:
        return np.array([0.0])

    f0 = sr / peak_lag
    if f0 < min_f0 or f0 > max_f0:
        f0 = 0.0

    harmonicity = autocorr[peak_lag] / max(autocorr[0], 1e-8)
    return np.array([f0, harmonicity], dtype=np.float32)


def extract_embeddings(audio_path, seg_dur=1.0, overlap=0.5):
    """
    从音频文件中提取 ECAPA-TDNN 嵌入序列

    参数:
        audio_path: 音频文件路径
        seg_dur: 片段时长(秒), 默认1.0s
        overlap: 重叠比例, 默认0.5

    返回:
        embeddings:  (N, 192) 每段的ECAPA嵌入
        segment_starts: (N,) 每段起始时间(秒)
        total_duration: float 总时长(秒)
    """
    classifier = _get_classifier()
    device = _get_device()

    # 加载音频
    sr, y = wavfile.read(audio_path)
    if y.dtype == np.int16:
        y = y.astype(np.float32) / 32768.0
    elif y.dtype == np.int32:
        y = y.astype(np.float32) / 2147483648.0
    else:
        y = y.astype(np.float32)
    if sr != 16000:
        y = resample(y, int(len(y) * 16000 / sr))
    if y.ndim > 1:
        y = y.mean(axis=1)

    total_dur = len(y) / 16000
    seg_len = int(seg_dur * 16000)
    hop_len = int(seg_len * (1 - overlap))
    if hop_len < 1:
        hop_len = seg_len

    n_seg = max(1, (len(y) - seg_len) // hop_len + 1)

    embeddings = []
    segment_starts = []

    for s in range(n_seg):
        start = s * hop_len
        end = min(start + seg_len, len(y))
        seg = y[start:end]
        if len(seg) < seg_len:
            seg = np.pad(seg, (0, seg_len - len(seg)))

        segment_starts.append(start / 16000)

        # ECAPA-TDNN 需要 (batch, time) 的 tensor
        seg_tensor = torch.from_numpy(seg).float().unsqueeze(0).to(device)

        with torch.no_grad():
            # encode_batch 返回 (batch, T, 192)
            emb = classifier.encode_batch(seg_tensor)
            # 在时间维度取平均 -> (batch, 192)
            emb = emb.mean(dim=1).squeeze(0)

        embeddings.append(emb.cpu().numpy())

    embeddings = np.array(embeddings)  # (N, 192)
    segment_starts = np.array(segment_starts)

    return embeddings, segment_starts, total_dur


def analyze_speakers(audio_path, seg_dur=1.0, overlap=0.5):
    """
    ECAPA-TDNN 说话人分析

    参数:
        audio_path: 音频文件路径
        seg_dur: 片段时长(秒), 默认1.0s
        overlap: 重叠比例, 默认0.5

    返回:
        dict 包含 speaker_count, speakers, timeline, silhouette_score 等
    """
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import silhouette_score
    from sklearn.cluster import KMeans

    # 1. 提取嵌入
    embeddings, segment_starts, total_dur = extract_embeddings(
        audio_path, seg_dur=seg_dur, overlap=overlap
    )
    n_seg = len(embeddings)

    # 2. 静音检测: RMS, 底部8%
    sr = 16000
    y_full = None
    try:
        _sr, _y = wavfile.read(audio_path)
        if _y.dtype == np.int16:
            _y = _y.astype(np.float32) / 32768.0
        elif _y.dtype == np.int32:
            _y = _y.astype(np.float32) / 2147483648.0
        else:
            _y = _y.astype(np.float32)
        if _sr != 16000:
            _y = resample(_y, int(len(_y) * 16000 / _sr))
        if _y.ndim > 1:
            _y = _y.mean(axis=1)
        y_full = _y
    except:
        pass

    seg_len = int(seg_dur * 16000)
    hop_len = int(seg_len * (1 - overlap))
    if hop_len < 1:
        hop_len = seg_len

    rms_values = np.zeros(n_seg)
    f0_features = np.zeros((n_seg, 2))

    for s in range(n_seg):
        start = s * hop_len
        end = min(start + seg_len, len(y_full) if y_full is not None else 16000)
        seg = y_full[start:end] if y_full is not None else np.zeros(seg_len)
        if len(seg) < seg_len:
            seg = np.pad(seg, (0, seg_len - len(seg)))

        rms_values[s] = np.sqrt(np.mean(seg ** 2) + 1e-10)
        f0_features[s] = extract_pitch(seg)

    rms_sorted = np.sort(rms_values)
    silence_th = rms_sorted[max(0, int(len(rms_sorted) * 0.08))]
    voice_mask = rms_values > silence_th
    n_voice = voice_mask.sum()

    voice_idx = [i for i, v in enumerate(voice_mask) if v]
    n_voice = len(voice_idx)

    if n_voice < 3:
        result = {
            'speaker_count': 0,
            'total_duration': round(total_dur, 1),
            'num_segments': n_seg,
            'voice_segments': n_voice,
            'silent_segments': n_seg - n_voice,
            'seg_dur': seg_dur,
            'overlap': overlap,
            'silhouette_score': 0.0,
            'speakers': [],
            'timeline': [(round(float(segment_starts[s]), 2), -1) for s in range(n_seg)],
            'silent_ranges': [],
        }
        return result

    # 3. GMM + BIC 自动确定说话人数（含 k=2 补正）
    embeddings_voice = embeddings[voice_mask]
    f0_voice = f0_features[voice_mask]

    scaler = StandardScaler()
    emb_scaled = scaler.fit_transform(embeddings_voice)

    from sklearn.mixture import GaussianMixture
    from sklearn.decomposition import PCA

    # PCA 降维避免高维 GMM 过拟合
    pca_dim = min(32, n_voice - 2, emb_scaled.shape[1])
    if pca_dim >= 2:
        pca = PCA(n_components=pca_dim)
        emb_reduced = pca.fit_transform(emb_scaled)
    else:
        emb_reduced = emb_scaled[:, :2]

    # GMM 拟合 k=1..max_k，BIC+AIC 联防
    max_k = min(8, n_voice // 3)
    bic_scores = {}
    aic_scores = {}
    for k in range(1, max_k + 1):
        gmm = GaussianMixture(n_components=k, covariance_type='diag',
                              n_init=3, max_iter=200, random_state=42)
        gmm.fit(emb_reduced)
        bic_scores[k] = gmm.bic(emb_reduced)
        aic_scores[k] = gmm.aic(emb_reduced)

    best_k = min(bic_scores, key=bic_scores.get)

    # AIC 联防: BIC 选 k=1 但 AIC 认为 k=2 更好 → k=2
    # （AIC 对 k=1→2 的差异比 BIC 敏感得多）
    if best_k == 1 and aic_scores[2] < aic_scores[1]:
        best_k = 2

    # 单说话人兜底（极低阈值，仅踢掉真正 100% 单人音频）
    if best_k == 2:
        k2_labels = KMeans(n_clusters=2, n_init=10, random_state=42).fit_predict(emb_scaled)
        if silhouette_score(emb_scaled, k2_labels) < 0.03:
            # 额外验证：两个簇的帧数比是否极不均衡（1 人模式）
            n0 = (k2_labels == 0).sum()
            n1 = (k2_labels == 1).sum()
            if min(n0, n1) / max(n0, n1) < 0.05:
                best_k = 1

    best_k = max(1, min(best_k, max_k))
    best_score = 0.0
    silhouette_scores = {2: 0.5}  # 占位

    # 用 GMM 做最终聚类（高维原始空间）
    gmm_final = GaussianMixture(n_components=best_k, covariance_type='diag',
                                n_init=5, max_iter=200, random_state=42)
    final_labels = gmm_final.fit_predict(emb_scaled)

    # 4. F0 性别估计
    raw_f0_means = np.array([
        f0_voice[final_labels == c, 0][f0_voice[final_labels == c, 0] > 0].mean()
        if (f0_voice[final_labels == c, 0] > 0).any() else 0
        for c in range(best_k)
    ])

    # 5. 后验合并: 基于embedding距离 + 时间重叠合并相近簇
    cluster_times = []
    for c in range(best_k):
        times = np.array([segment_starts[i] for i in range(len(final_labels))
                          if final_labels[i] == c])
        cluster_times.append(times)

    # 计算每个簇的embedding质心
    cluster_centroids = []
    for c in range(best_k):
        mask = final_labels == c
        if mask.sum() > 0:
            cluster_centroids.append(emb_scaled[mask].mean(axis=0))
        else:
            cluster_centroids.append(np.zeros(emb_scaled.shape[1]))

    if best_k >= 2:
        def time_overlap_pct(times_a, times_b):
            if len(times_a) == 0 or len(times_b) == 0:
                return 0.0
            overlap_count = 0
            for ta in times_a:
                for tb in times_b:
                    if abs(ta - tb) < seg_dur * 0.75:
                        overlap_count += 1
                        break
            return overlap_count / min(len(times_a), len(times_b))

        # 计算质心距离阈值
        from scipy.spatial.distance import cdist as centroid_dist
        dists = centroid_dist(cluster_centroids, cluster_centroids)
        np.fill_diagonal(dists, np.inf)
        median_dist = np.median(dists[dists < np.inf]) if np.any(dists < np.inf) else 1.0

        merged = list(range(best_k))
        for i in range(best_k):
            for j in range(i+1, best_k):
                if merged[i] == merged[j]:
                    continue
                emb_dist = np.linalg.norm(cluster_centroids[i] - cluster_centroids[j])
                t_overlap = time_overlap_pct(cluster_times[i], cluster_times[j])
                
                # 判断性别是否相同
                f0_i = raw_f0_means[i] if raw_f0_means[i] > 0 else 0
                f0_j = raw_f0_means[j] if raw_f0_means[j] > 0 else 0
                same_gender = ((f0_i > 160 and f0_j > 160) or (f0_i <= 160 and f0_j <= 160))
                
                should_merge = False
                # 条件1: 不同时段但embedding很近 + 同性别 → 同一人
                if emb_dist < median_dist * 0.7 and same_gender and t_overlap < 0.1:
                    should_merge = True
                # 条件2: 同时段重叠embedding近 → 可能是被错误分开的
                if emb_dist < median_dist * 0.5 and t_overlap > 0.3:
                    should_merge = True
                # 条件3: embedding非常近（远小于中位数）
                if emb_dist < median_dist * 0.35:
                    should_merge = True
                    
                if should_merge:
                    old = merged[j]
                    for m in range(best_k):
                        if merged[m] == old:
                            merged[m] = merged[i]

        unique_m = sorted(set(merged))
        if len(unique_m) < best_k:
            remap = {m: new_id for new_id, m in enumerate(unique_m)}
            for i in range(len(final_labels)):
                final_labels[i] = remap[merged[final_labels[i]]]
            best_k = len(unique_m)

    # 6. 构建完整时间线
    full_labels = -np.ones(n_seg, dtype=np.int32)
    for vi, orig_idx in enumerate(voice_idx):
        full_labels[orig_idx] = final_labels[vi]

    timeline = [(round(float(segment_starts[s]), 2), int(full_labels[s]))
                for s in range(n_seg)]

    # 7. 构建说话人详情
    speakers = []
    for spk_id in sorted(set(final_labels)):
        voice_segs = [vi for vi in range(n_voice) if final_labels[vi] == spk_id]
        orig_segs = [voice_idx[vi] for vi in voice_segs]
        pct = len(orig_segs) / n_voice * 100

        spk_f0 = f0_voice[voice_segs, 0]
        spk_f0_valid = spk_f0[spk_f0 > 0]
        avg_f0 = float(spk_f0_valid.mean()) if len(spk_f0_valid) > 0 else 0

        if avg_f0 > 160:
            gender_hint = "女声"
        elif avg_f0 > 80:
            gender_hint = "男声"
        else:
            gender_hint = "未知"

        # 合并连续片段
        active_ranges = []
        if orig_segs:
            start_idx = orig_segs[0]
            end_idx = orig_segs[0]
            for k in range(1, len(orig_segs)):
                if orig_segs[k] == end_idx + 1:
                    end_idx = orig_segs[k]
                else:
                    t_start = segment_starts[start_idx]
                    t_end = segment_starts[min(end_idx, n_seg-1)] + seg_dur
                    active_ranges.append((round(float(t_start), 1), round(float(t_end), 1)))
                    start_idx = end_idx = orig_segs[k]
            t_start = segment_starts[start_idx]
            t_end = segment_starts[min(end_idx, n_seg-1)] + seg_dur
            active_ranges.append((round(float(t_start), 1), round(float(t_end), 1)))

        speakers.append({
            'id': int(spk_id),
            'segments': len(orig_segs),
            'percentage': round(pct, 1),
            'avg_f0': round(avg_f0, 1),
            'gender_hint': gender_hint,
            'active_ranges': active_ranges
        })

    # 静音段时间段
    silent_ranges = []
    silent_segs = [s for s in range(n_seg) if full_labels[s] == -1]
    if silent_segs:
        s_start = silent_segs[0]
        s_end = silent_segs[0]
        for k in range(1, len(silent_segs)):
            if silent_segs[k] == s_end + 1:
                s_end = silent_segs[k]
            else:
                silent_ranges.append((round(float(segment_starts[s_start]), 1),
                                      round(float(segment_starts[min(s_end, n_seg-1)] + seg_dur), 1)))
                s_start = s_end = silent_segs[k]
        silent_ranges.append((round(float(segment_starts[s_start]), 1),
                              round(float(segment_starts[min(s_end, n_seg-1)] + seg_dur), 1)))

    return {
        'speaker_count': best_k,
        'total_duration': round(total_dur, 1),
        'num_segments': n_seg,
        'voice_segments': n_voice,
        'silent_segments': n_seg - n_voice,
        'silent_ranges': silent_ranges,
        'seg_dur': seg_dur,
        'overlap': overlap,
        'silhouette_score': round(float(best_score), 4),
        'speakers': speakers,
        'timeline': timeline
    }


def print_timeline(result):
    """控制台打印分析结果"""
    print(f"\n{'='*60}")
    print(f"  ECAPA-TDNN 说话人分析结果")
    print(f"  片段={result['seg_dur']}s, 重叠={result['overlap']}")
    print(f"{'='*60}")
    print(f"  音频时长: {result['total_duration']:.1f}秒")
    nv = result.get('voice_segments', result['num_segments'])
    ns = result.get('silent_segments', 0)
    print(f"  分析片段: {result['num_segments']}个 (有声:{nv} 静音:{ns})")
    print(f"  说话人数: {result['speaker_count']}人")
    print(f"  轮廓系数: {result['silhouette_score']}")
    print(f"  Powered by ECAPA-TDNN (VoxCeleb pretrained)")
    for spk in sorted(result['speakers'], key=lambda x: -x['percentage']):
        print(f"\n  [说话人{spk['id']+1}] {spk['segments']}片段 ({spk['percentage']:.1f}%)")
        print(f"    估计性别: {spk['gender_hint']} (F0={spk['avg_f0']:.0f}Hz)")
        for start, end in spk['active_ranges']:
            print(f"    {start:.1f}s -- {end:.1f}s")
    sr = result.get('silent_ranges', [])
    if sr:
        print(f"\n  [静音段]")
        for start, end in sr:
            print(f"    {start:.1f}s -- {end:.1f}s")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: python ecapa_speaker.py <音频文件>")
        print("示例: python ecapa_speaker.py ../testvoices/tex.wav")
        sys.exit(1)

    audio_file = sys.argv[1]
    if not os.path.exists(audio_file):
        print(f"文件不存在: {audio_file}")
        sys.exit(1)

    print(f"分析: {audio_file}")
    print("正在加载 ECAPA-TDNN 模型...")
    result = analyze_speakers(audio_file)
    print_timeline(result)
