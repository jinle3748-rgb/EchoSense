"""
说话人时间线分析程序 (增强版)
- 0.5s 重叠滑动窗口 (50% overlap)，更细粒度
- 音高(F0)性别辅助特征，区分男女
- 多聚类投票 (KMeans + Agglomerative + DBSCAN)
- 中值滤波平滑
- 独立运行: python speaker_timeline.py <音频文件>
- GUI调用: from ModelTrain.speaker_timeline import analyze_speakers
"""

import os, sys, pickle, time, warnings
warnings.filterwarnings('ignore')
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from scipy.io import wavfile

# ── MFCC Transform (shared, CPU) ──
MFCC_TRANSFORM = torchaudio.transforms.MFCC(
    sample_rate=16000, n_mfcc=20,
    melkwargs={'n_fft': 512, 'hop_length': 160, 'n_mels': 20,
               'power': 2.0, 'window_fn': torch.hamming_window}
)
from scipy.signal import resample, medfilt, butter, filtfilt
from collections import Counter

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models", "XVector_full")

# ── XVector Model (matches train_final.py architecture) ──
class TDNNBlock(nn.Module):
    def __init__(self, i, o, k, d):
        super().__init__()
        self.c = nn.Conv1d(i, o, k, dilation=d, padding='same')
        self.b = nn.BatchNorm1d(o); self.r = nn.ReLU()
    def forward(self, x): return self.r(self.b(self.c(x)))

class StatPool(nn.Module):
    def forward(self, x): return torch.cat([x.mean(2), x.std(2)], 1)

class XVector(nn.Module):
    def __init__(self, n_mfcc=20, n_classes=10, emb_dim=256):
        super().__init__()
        self.t = nn.Sequential(
            TDNNBlock(n_mfcc,512,5,1), TDNNBlock(512,512,3,2),
            TDNNBlock(512,512,3,3), TDNNBlock(512,512,1,1),
            TDNNBlock(512,1500,1,1), StatPool(),
            nn.Linear(3000,512), nn.ReLU(), nn.Linear(512,512), nn.ReLU(),
            nn.Linear(512,emb_dim)
        )
        self.cls = nn.Linear(emb_dim, n_classes)
        self.emb_dim = emb_dim
    def forward(self, x, return_emb=False):
        emb = self.t(x.transpose(1,2))
        if return_emb: return emb
        return self.cls(emb)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ── MFCC Extraction ──
def extract_mfcc(waveform, sr=16000):
    """(1, samples) tensor → (1, n_mfcc, T)"""
    if sr != 16000:
        resampler = torchaudio.transforms.Resample(sr, 16000)
        waveform = resampler(waveform)
    return MFCC_TRANSFORM(waveform)

def load_model():
    meta = pickle.load(open(os.path.join(MODEL_DIR, "xvector_torch_meta.pkl"), 'rb'))
    model = XVector(n_mfcc=20, n_classes=meta['n_speakers'],
                     emb_dim=meta['emb_dim']).to(DEVICE)
    model.load_state_dict(torch.load(os.path.join(MODEL_DIR, "best_model.pt"),
                         map_location=DEVICE))
    model.eval()
    return model, meta


def extract_pitch(y, sr=16000, min_f0=80, max_f0=400):
    """
    提取基频 F0 (音高)，用于辅助性别判断
    男性: 80-160Hz, 女性: 160-400Hz
    """
    from scipy.signal import correlate

    # 带通滤波
    nyq = sr / 2
    b, a = butter(4, [min_f0/nyq, min(max_f0/nyq, 0.95)], btype='band')
    y_filt = filtfilt(b, a, y)

    # 自相关法估计基频
    n = len(y_filt)
    if n < sr // min_f0:
        return np.array([0.0])

    autocorr = correlate(y_filt, y_filt, mode='full')
    autocorr = autocorr[n-1:]  # 取正延迟

    # 搜索范围: min_f0 ~ max_f0 对应的延迟
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

    # 也返回 harmonicity (自相关峰值强度)
    harmonicity = autocorr[peak_lag] / max(autocorr[0], 1e-8)
    return np.array([f0, harmonicity], dtype=np.float32)


def analyze_speakers(audio_path, seg_dur=0.5, overlap=0.5):
    """
    增强版说话人分析

    参数:
        audio_path: 音频文件路径
        seg_dur: 片段时长(秒), 默认0.5s
        overlap: 重叠比例, 默认0.5
    """
    model, meta = load_model()

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
    hop_len = int(seg_len * (1 - overlap))  # 50% overlap -> hop = seg_len/2
    if hop_len < 1:
        hop_len = seg_len

    # 提取片段特征: XVector嵌入 + 音高 + 能量
    embeddings = []
    f0_features = []
    segment_starts = []
    rms_values = []  # 每段RMS

    n_seg = max(1, (len(y) - seg_len) // hop_len + 1)
    for s in range(n_seg):
        start = s * hop_len
        end = min(start + seg_len, len(y))
        seg = y[start:end]
        if len(seg) < seg_len:
            seg = np.pad(seg, (0, seg_len - len(seg)))

        # RMS
        rms = np.sqrt(np.mean(seg ** 2) + 1e-10)
        rms_values.append(rms)
        segment_starts.append(start / 16000)

        # XVector嵌入
        seg_tensor = torch.from_numpy(seg).float().unsqueeze(0)
        mfcc = extract_mfcc(seg_tensor).transpose(1, 2)
        with torch.no_grad():
            emb = model(mfcc.to(DEVICE), return_emb=True)
        embeddings.append(emb.cpu().numpy().flatten())

        # 音高特征
        f0feat = extract_pitch(seg)
        f0_features.append(f0feat)

    embeddings = np.array(embeddings)
    f0_features = np.array(f0_features)
    rms_values = np.array(rms_values)

    # 自适应静音阈值: 底部10%分位
    rms_sorted = np.sort(rms_values)
    silence_th = rms_sorted[max(0, int(len(rms_sorted) * 0.08))]  # 最安静8% → 静音
    voice_mask = rms_values > silence_th
    n_voice = voice_mask.sum()
    n_silent = n_seg - n_voice

    # 有声段索引映射: voice_idx[i] = 原始segment索引
    voice_idx = [i for i, v in enumerate(voice_mask) if v]
    n_voice = len(voice_idx)

    # 只取有声段做聚类
    embeddings_voice = embeddings[voice_mask]
    f0_features_voice = f0_features[voice_mask]

    if n_voice < 3:
        return {
            'speaker_count': 0,
            'total_duration': round(total_dur, 1),
            'num_segments': n_seg,
            'voice_segments': n_voice,
            'silent_segments': n_seg - n_voice,
            'seg_dur': seg_dur,
            'overlap': overlap,
            'silhouette_score': 0.0,
            'speakers': [],
            'timeline': [(round(segment_starts[s], 2), -1) for s in range(n_seg)]
        }

    # 合并特征: XVector(256) + F0(2) = 258维
    from sklearn.preprocessing import StandardScaler
    scaler_emb = StandardScaler()
    emb_scaled = scaler_emb.fit_transform(embeddings_voice)

    # F0特征也标准化（给较小权重，避免主导聚类）
    f0_scaled = np.zeros_like(f0_features_voice)
    for j in range(f0_features_voice.shape[1]):
        col = f0_features_voice[:, j]
        if col.std() > 0:
            f0_scaled[:, j] = (col - col.mean()) / col.std()

    # XVector权重 >> F0权重 (F0只是辅助)
    combined = np.concatenate([emb_scaled * 1.0, f0_scaled * 0.3], axis=1)

    # ============================================
    # 多聚类方法 + 惩罚 + 后验合并
    # ============================================
    from sklearn.cluster import AgglomerativeClustering, KMeans, DBSCAN
    from sklearn.metrics import silhouette_score

    max_k = min(12, n_voice // 10)
    k_range = range(2, max_k + 1)

    # 方法1: AgglomerativeClustering — 间隙阈值法
    scores1_raw = {}
    for k in k_range:
        try:
            lbl = AgglomerativeClustering(n_clusters=k).fit_predict(combined)
            s = silhouette_score(combined, lbl)
            scores1_raw[k] = s
        except:
            scores1_raw[k] = -1

    # 方法2: KMeans
    scores2_raw = {}
    for k in k_range:
        try:
            s_best = -1
            for _ in range(20):
                lbl = KMeans(n_clusters=k, random_state=np.random.randint(9999),
                            n_init=1).fit_predict(combined)
                s = silhouette_score(combined, lbl)
                if s > s_best:
                    s_best = s
            scores2_raw[k] = s_best
        except:
            scores2_raw[k] = -1

    # 选择k: 取两种方法平均轮廓。从k=2开始，只在下一k提升>0.012时递进
    avg_scores = {}
    for k in k_range:
        a = scores1_raw.get(k, -1)
        b = scores2_raw.get(k, -1)
        if a > -1 and b > -1:
            avg_scores[k] = (a + b) / 2
        elif a > -1:
            avg_scores[k] = a
        elif b > -1:
            avg_scores[k] = b

    best_k1 = max(scores1_raw, key=scores1_raw.get) if scores1_raw else 1
    best_k2 = max(scores2_raw, key=scores2_raw.get) if scores2_raw else 1

    # 间隙法定k
    best_k = 2
    if 2 in avg_scores:
        for k in range(3, max_k + 1):
            if k in avg_scores and avg_scores[k] > avg_scores.get(k-1, -99) + 0.012:
                best_k = k

    # 方法3: DBSCAN
    best_k3, s3 = 1, -1
    try:
        from sklearn.neighbors import NearestNeighbors
        nn = NearestNeighbors(n_neighbors=min(5, len(embeddings)-1))
        nn.fit(combined[:200])  # 采样加速
        dists = np.sort(nn.kneighbors(combined[:200])[0][:, -1])
        if len(dists) > 2:
            diffs = np.diff(dists)
            eps = dists[np.argmax(diffs) + 1] * 0.75
        else:
            eps = 0.5

        db_lbl = DBSCAN(eps=max(eps, 0.3),
                        min_samples=max(3, len(embeddings)//15)).fit_predict(combined)
        n_db = len(set(db_lbl)) - (1 if -1 in db_lbl else 0)
        if n_db >= 2:
            if -1 in db_lbl:
                noise_idx = np.where(db_lbl == -1)[0]
                for ni in noise_idx:
                    min_d = float('inf')
                    min_c = 0
                    for c in range(n_db):
                        center = combined[db_lbl == c].mean(axis=0)
                        d = np.linalg.norm(combined[ni] - center)
                        if d < min_d:
                            min_d = d; min_c = c
                    db_lbl[ni] = min_c
            best_k3 = n_db
            s3 = silhouette_score(combined, db_lbl)
    except:
        pass

    # 取中位数k (保守)
    k_candidates = []
    if best_k1 > 1: k_candidates.append(best_k1)
    if best_k2 > 1: k_candidates.append(best_k2)
    if best_k3 > 1: k_candidates.append(best_k3)
    k_candidates.append(best_k)  # 间隙法结果

    if k_candidates:
        best_k = int(np.median(k_candidates))

    best_k = max(2, min(best_k, max_k))
    best_score = max(avg_scores.get(best_k, -1), s3, 0.0)

    # 最终聚类 (KMeans 多初始化)
    kmeans = KMeans(n_clusters=best_k, n_init=30, random_state=42)
    final_labels = kmeans.fit_predict(combined)

    # ============================================
    # 后验合并: 同性别且时间基本不重叠 → 很可能是同一人
    # ============================================
    centroids = np.array([combined[final_labels == c].mean(axis=0) for c in range(best_k)])
    raw_f0_means = np.array([f0_features_voice[final_labels == c, 0][f0_features_voice[final_labels == c, 0] > 0].mean()
                              if (f0_features_voice[final_labels == c, 0] > 0).any() else 0
                              for c in range(best_k)])

    cluster_times = []
    for c in range(best_k):
        times = np.array([segment_starts[i] for i in range(len(final_labels)) if final_labels[i] == c])
        cluster_times.append(times)

    from scipy.spatial.distance import pdist, squareform
    cdist_matrix = squareform(pdist(centroids))
    all_dists = cdist_matrix[cdist_matrix > 0]
    avg_dist = all_dists.mean() if (all_dists > 0).any() else 1.0

    if best_k >= 2:
        def time_overlap_pct(times_a, times_b):
            if len(times_a) == 0 or len(times_b) == 0:
                return 0.0
            overlap_count = 0
            for ta in times_a:
                for tb in times_b:
                    if abs(ta - tb) < 0.75:
                        overlap_count += 1
                        break
            return overlap_count / min(len(times_a), len(times_b))

        def gender(f0):
            return 'F' if f0 > 155 else 'M'

        merged = list(range(best_k))
        for i in range(best_k):
            for j in range(i+1, best_k):
                if merged[i] == merged[j]:
                    continue
                overlap = time_overlap_pct(cluster_times[i], cluster_times[j])
                emb_dist = cdist_matrix[i, j]

                # 合并条件 (按优先级):
                # 1) 两人几乎不同时出现(<3%重叠) → 同一人
                # 2) 嵌入距离 < 平均*0.6 → 同一人
                # 3) 同性别 且 重叠<20% → 同一人
                should_merge = False
                if overlap < 0.03:
                    should_merge = True
                elif emb_dist < avg_dist * 0.6:
                    should_merge = True
                elif gender(raw_f0_means[i]) == gender(raw_f0_means[j]) and overlap < 0.20:
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

    # ============================================
    # 中值滤波平滑 (减少快速切换)
    # ============================================
    if len(final_labels) >= 3:
        final_labels = medfilt(final_labels.astype(np.float32), kernel_size=3).astype(np.int32)

    # 合并短孤立段 (< 2个连续片段)
    min_run = 2
    cleaned = final_labels.copy()
    i = 0
    while i < len(cleaned):
        j = i
        while j < len(cleaned) and cleaned[j] == cleaned[i]:
            j += 1
        run_len = j - i
        if run_len < min_run and i > 0:
            cleaned[i:j] = cleaned[i-1]
        elif run_len < min_run and j < len(cleaned):
            cleaned[i:j] = cleaned[j]
        i = j

    final_labels = cleaned

    # 构建完整时间线 (含静音段)
    # 先创建全标签数组: -1=静音, 0...=说话人
    full_labels = -np.ones(n_seg, dtype=np.int32)
    for vi, orig_idx in enumerate(voice_idx):
        full_labels[orig_idx] = final_labels[vi]

    timeline = [(round(segment_starts[s], 2), int(full_labels[s]))
                for s in range(n_seg)]

    # 构建说话人详情
    speakers = []
    for spk_id in sorted(set(final_labels)):
        # voice段中的索引
        voice_segs = [vi for vi in range(n_voice) if final_labels[vi] == spk_id]
        # 映射回原始段索引
        orig_segs = [voice_idx[vi] for vi in voice_segs]
        pct = len(orig_segs) / n_voice * 100

        # 该说话人的平均F0
        spk_f0 = f0_features_voice[voice_segs, 0]
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
                    active_ranges.append((round(t_start, 1), round(t_end, 1)))
                    start_idx = end_idx = orig_segs[k]
            t_start = segment_starts[start_idx]
            t_end = segment_starts[min(end_idx, n_seg-1)] + seg_dur
            active_ranges.append((round(t_start, 1), round(t_end, 1)))

        speakers.append({
            'id': int(spk_id),
            'segments': len(orig_segs),
            'percentage': round(pct, 1),
            'avg_f0': round(avg_f0, 1),
            'gender_hint': gender_hint,
            'active_ranges': active_ranges
        })

    # 找出静音段时间段
    silent_ranges = []
    silent_segs = [s for s in range(n_seg) if full_labels[s] == -1]
    if silent_segs:
        s_start = silent_segs[0]
        s_end = silent_segs[0]
        for k in range(1, len(silent_segs)):
            if silent_segs[k] == s_end + 1:
                s_end = silent_segs[k]
            else:
                silent_ranges.append((round(segment_starts[s_start], 1),
                                      round(segment_starts[min(s_end, n_seg-1)] + seg_dur, 1)))
                s_start = s_end = silent_segs[k]
        silent_ranges.append((round(segment_starts[s_start], 1),
                              round(segment_starts[min(s_end, n_seg-1)] + seg_dur, 1)))

    return {
        'speaker_count': best_k,
        'total_duration': round(total_dur, 1),
        'num_segments': n_seg,
        'voice_segments': n_voice,
        'silent_segments': n_seg - n_voice,
        'silent_ranges': silent_ranges,
        'seg_dur': seg_dur,
        'overlap': overlap,
        'silhouette_score': round(best_score, 4),
        'speakers': speakers,
        'timeline': timeline
    }


def print_timeline(result):
    print(f"\n{'='*60}")
    print(f"  说话人分析结果 (增强版, 片段={result['seg_dur']}s, 重叠={result['overlap']})")
    print(f"{'='*60}")
    print(f"  音频时长: {result['total_duration']:.1f}秒")
    nv = result.get('voice_segments', result['num_segments'])
    ns = result.get('silent_segments', 0)
    print(f"  分析片段: {result['num_segments']}个 (有声:{nv} 静音:{ns})")
    print(f"  说话人数: {result['speaker_count']}人")
    print(f"  轮廓系数: {result['silhouette_score']}")
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


def plot_timeline(result, save_path=None):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6),
                                    gridspec_kw={'height_ratios': [1, 3]})

    speakers = sorted(result['speakers'], key=lambda x: x['id'])
    colors = ['#66c2a5', '#fc8d62', '#8da0cb', '#e78ac3', '#a6d854',
              '#ffd92f', '#e5c494', '#b3b3b3']

    # 饼图
    labels = [f"{s['gender_hint']}\n说话人{s['id']+1}" for s in speakers]
    sizes = [s['percentage'] for s in speakers]
    ax1.pie(sizes, labels=labels, autopct='%1.1f%%',
            colors=[colors[s['id']%len(colors)] for s in speakers], startangle=90)
    ax1.set_title(f"说话人分布 (共{result['speaker_count']}人)")

    # 时间线
    for spk in speakers:
        for start, end in spk['active_ranges']:
            ax2.barh(spk['id'], end - start, left=start, height=0.7,
                     color=colors[spk['id'] % len(colors)],
                     edgecolor='white', linewidth=0.5)

    ax2.set_yticks([s['id'] for s in speakers])
    ax2.set_yticklabels([f"说话人{s['id']+1} ({s['gender_hint']})" for s in speakers])
    ax2.set_xlabel("时间 (秒)")
    ax2.set_title(f"说话人活动时间线")
    ax2.set_xlim(0, result['total_duration'])
    ax2.grid(axis='x', alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        plt.close()
        return save_path
    else:
        plt.show()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: python speaker_timeline.py <音频文件>")
        print("示例: python speaker_timeline.py ../testvoices/tex.wav")
        sys.exit(1)

    audio_file = sys.argv[1]
    if not os.path.exists(audio_file):
        print(f"文件不存在: {audio_file}")
        sys.exit(1)

    print(f"分析: {audio_file}")
    result = analyze_speakers(audio_file)
    print_timeline(result)

    out_png = os.path.splitext(audio_file)[0] + "_speakers_v2.png"
    plot_timeline(result, out_png)
    print(f"时间线图: {out_png}")
