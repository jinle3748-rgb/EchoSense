"""
XVector训练 V2 - 预提取MFCC + GPU训练
1. 遍历所有wav，提取MFCC特征并缓存为.npy
2. 从缓存加载训练集，GPU快速训练
"""
import os, sys, time, pickle, glob, warnings, json
warnings.filterwarnings('ignore')

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchaudio
from scipy.io import wavfile
from scipy.signal import resample

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CACHE_DIR = os.path.join(BASE_DIR, 'models', 'XVector_full', 'mfcc_cache')
DATA_DIR = os.path.join(BASE_DIR, 'RecogizeTrain', 'data', 'raw', 'data_aishell', 'wav', 'train')
MODEL_DIR = os.path.join(BASE_DIR, 'models', 'XVector_full')

print(f"[DEVICE] {DEVICE}", flush=True)
if torch.cuda.is_available():
    print(f"[GPU] {torch.cuda.get_device_name(0)}", flush=True)

# =====================================================
# MFCC Transform (共享，GPU上更慢，固定CPU)
# =====================================================
MFCC_TRANSFORM = torchaudio.transforms.MFCC(
    sample_rate=16000, n_mfcc=20,
    melkwargs={'n_fft': 512, 'hop_length': 160, 'n_mels': 20,
               'power': 2.0, 'window_fn': torch.hamming_window}
)

def extract_mfcc(waveform, sr=16000):
    """兼容speaker_timeline.py: 输入 (1, samples) tensor → (1, n_mfcc, T)"""
    if sr != 16000:
        resampler = torchaudio.transforms.Resample(sr, 16000)
        waveform = resampler(waveform)
    return MFCC_TRANSFORM(waveform)

def extract_mfcc_from_wav(wav_path, seg_dur=2.0):
    """从wav文件提取MFCC序列，返回list of (T, 20) tensors"""
    try:
        sr, y = wavfile.read(wav_path)
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

        seg_len = int(seg_dur * 16000)
        n_seg = max(1, len(y) // seg_len)
        segments = []

        for s in range(n_seg):
            seg = y[s*seg_len:(s+1)*seg_len]
            if len(seg) < seg_len:
                seg = np.pad(seg, (0, seg_len - len(seg)))
            seg_tensor = torch.from_numpy(seg).float().unsqueeze(0)
            mfcc = MFCC_TRANSFORM(seg_tensor)  # (1, 20, T)
            segments.append(mfcc.squeeze(0).transpose(0, 1).numpy())  # (T, 20)

        return segments
    except Exception as e:
        return None


# =====================================================
# XVector Model
# =====================================================
class TDNNBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, dilation=1):
        super().__init__()
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size, dilation=dilation,
                              padding=(kernel_size-1)*dilation//2)
        self.bn = nn.BatchNorm1d(out_ch)
        self.relu = nn.ReLU()
    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

class StatisticsPooling(nn.Module):
    def forward(self, x):
        mean = x.mean(dim=2)
        std = x.std(dim=2)
        return torch.cat([mean, std], dim=1)

class XVector(nn.Module):
    def __init__(self, n_mfcc=20, n_classes=10, emb_dim=256):
        super().__init__()
        self.tdnn1 = TDNNBlock(n_mfcc, 512, 5)
        self.tdnn2 = TDNNBlock(512, 512, 3)
        self.tdnn3 = TDNNBlock(512, 512, 3)
        self.tdnn4 = TDNNBlock(512, 512, 1)
        self.tdnn5 = TDNNBlock(512, 1500, 1)
        self.pool = StatisticsPooling()
        self.fc1 = nn.Linear(3000, emb_dim)
        self.bn1 = nn.BatchNorm1d(emb_dim)
        self.fc2 = nn.Linear(emb_dim, n_classes)
        self.emb_dim = emb_dim
        self.dropout = nn.Dropout(0.3)

    def forward(self, x, return_emb=False):
        x = x.transpose(1, 2)
        x = self.tdnn1(x)
        x = self.dropout(x)
        x = self.tdnn2(x)
        x = self.dropout(x)
        x = self.tdnn3(x)
        x = self.dropout(x)
        x = self.tdnn4(x)
        x = self.tdnn5(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        emb = F.relu(self.bn1(self.fc1(x)))
        if return_emb:
            return emb
        return self.fc2(emb)


# =====================================================
# Phase 1: Preprocess (extract MFCC, cache to disk)
# =====================================================
class CachedDataset(Dataset):
    """从缓存的.npy文件加载MFCC特征的数据集"""
    def __init__(self, cache_files, labels, c2i):
        self.cache_files = cache_files
        self.labels = [c2i[l] for l in labels]
        self.c2i = c2i

    def __len__(self):
        return len(self.cache_files)

    def __getitem__(self, idx):
        mfcc = torch.from_numpy(np.load(self.cache_files[idx])).float()
        return mfcc, self.labels[idx]


def collate_fn(batch):
    features, labels = zip(*batch)
    max_len = max(f.shape[0] for f in features)
    padded = []
    for f in features:
        if f.shape[0] < max_len:
            pad = torch.zeros(max_len - f.shape[0], f.shape[1])
            f = torch.cat([f, pad], dim=0)
        padded.append(f)
    return torch.stack(padded), torch.tensor(labels)


def preprocess_and_cache():
    """遍历所有wav，提取MFCC，缓存"""
    os.makedirs(CACHE_DIR, exist_ok=True)

    # 收集所有文件
    af, lb = [], []
    for d in sorted(os.listdir(DATA_DIR)):
        dp = os.path.join(DATA_DIR, d)
        if os.path.isdir(dp):
            for w in glob.glob(os.path.join(dp, "*.wav")):
                af.append(w)
                lb.append(d)

    n_speakers = len(set(lb))
    total_files = len(af)
    log_print(f"[预提取] {n_speakers}说话人, {total_files}文件")

    # 检查已缓存的文件(缓存名含_sN段号, 需提取原始wav名)
    existing = set()
    for f in glob.glob(os.path.join(CACHE_DIR, "*.npy")):
        name = os.path.basename(f).replace('.npy', '')
        parts = name.rsplit('_s', 1)
        if len(parts) == 2 and parts[1].isdigit():
            existing.add(parts[0])

    # 只处理未缓存的文件
    todo = [(path, lbl) for path, lbl in zip(af, lb) 
            if os.path.basename(path).replace('.wav', '') not in existing]
    
    skipped = total_files - len(todo)
    log_print(f"[预提取] 已缓存: {skipped}, 需处理: {len(todo)}")

    # 构建 base名→说话人 映射 (一次性, 避免后续O(n)遍历)
    base_to_label = {}
    for path, lbl in zip(af, lb):
        base_to_label[os.path.basename(path).replace('.wav', '')] = lbl

    cache_map = {}  # cache_file -> label
    for i, (path, lbl) in enumerate(todo):
        if (i + 1) % 5000 == 0 or i == 0:
            print(f"  进度: {i+1}/{len(todo)} ({100*(i+1)/len(todo):.1f}%)", flush=True)

        segments = extract_mfcc_from_wav(path)
        if segments is None:
            continue

        base = os.path.basename(path).replace('.wav', '')
        for seg_idx, seg in enumerate(segments):
            cache_file = os.path.join(CACHE_DIR, f"{base}_s{seg_idx}.npy")
            np.save(cache_file, seg)
            cache_map[cache_file] = lbl

    # 直接遍历npy文件构建cache_map (比遍历wav+glob快100倍)
    for f in glob.glob(os.path.join(CACHE_DIR, "*.npy")):
        name = os.path.basename(f).replace('.npy', '')
        parts = name.rsplit('_s', 1)
        if len(parts) == 2 and parts[1].isdigit():
            base = parts[0]
            if base in base_to_label and f not in cache_map:
                cache_map[f] = base_to_label[base]

    log_print(f"[预提取] 完成! 缓存片段: {len(cache_map)}")
    return cache_map


# =====================================================
# Phase 2: Training
# =====================================================
# 日志工具：同时输出到终端和文件，立即刷新
LOG_FILE = os.path.join(os.path.dirname(BASE_DIR), 'train_output.txt')
_log_fh = None

def log_print(*args, **kwargs):
    """print同时写入日志文件并flush"""
    msg = ' '.join(str(a) for a in args)
    print(msg, flush=True, **kwargs)
    global _log_fh
    if _log_fh is None:
        _log_fh = open(LOG_FILE, 'w', encoding='utf-8')
    _log_fh.write(msg + '\n')
    _log_fh.flush()


def train():
    import random
    global _log_fh
    _log_fh = open(LOG_FILE, 'w', encoding='utf-8')
    log_print("\n[Phase 1] 预提取MFCC特征...")
    t0 = time.time()
    cache_map = preprocess_and_cache()
    t1 = time.time()
    log_print(f"[Phase 1] 耗时: {t1-t0:.1f}s")

    cache_files = list(cache_map.keys())
    cache_labels = list(cache_map.values())
    classes = sorted(set(cache_labels))
    n_speakers = len(classes)
    c2i = {c: i for i, c in enumerate(classes)}
    log_print(f"[数据] {len(cache_files)}片段, {n_speakers}说话人")

    # 划分
    np.random.seed(42)
    idx = np.random.permutation(len(cache_files))
    sp = int(len(cache_files) * 0.8)
    train_files = [cache_files[i] for i in idx[:sp]]
    train_labels = [c2i[cache_labels[i]] for i in idx[:sp]]
    val_files = [cache_files[i] for i in idx[sp:]]
    val_labels = [c2i[cache_labels[i]] for i in idx[sp:]]
    log_print(f"[划分] 训练: {len(train_files)}, 验证: {len(val_files)}")

    # 预加载所有数据到内存 (避免训练时磁盘I/O)
    log_print("[预加载] 训练集到内存...")
    train_data, skipped = [], 0
    for i, f in enumerate(train_files):
        if (i + 1) % 10000 == 0:
            log_print(f"  训练: {i+1}/{len(train_files)} ({100*(i+1)/len(train_files):.0f}%)")
        try:
            train_data.append((torch.from_numpy(np.load(f)).float(), train_labels[i]))
        except Exception:
            skipped += 1
            if skipped <= 3:
                log_print(f"  [跳过损坏] {os.path.basename(f)}")
    if skipped:
        log_print(f"[预加载] 训练集完成: {len(train_data)} (跳过{skipped}个损坏)")
    else:
        log_print(f"[预加载] 训练集完成: {len(train_data)}")

    log_print("[预加载] 验证集到内存...")
    val_data, skipped = [], 0
    for i, f in enumerate(val_files):
        if (i + 1) % 10000 == 0:
            log_print(f"  验证: {i+1}/{len(val_files)} ({100*(i+1)/len(val_files):.0f}%)")
        try:
            val_data.append((torch.from_numpy(np.load(f)).float(), val_labels[i]))
        except Exception:
            skipped += 1
            if skipped <= 3:
                log_print(f"  [跳过损坏] {os.path.basename(f)}")
    if skipped:
        log_print(f"[预加载] 验证集完成: {len(val_data)} (跳过{skipped}个损坏)")
    else:
        log_print(f"[预加载] 验证集完成: {len(val_data)}")

    # 模型
    model = XVector(n_mfcc=20, n_classes=n_speakers, emb_dim=256).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    log_print(f"[模型] 参数: {n_params:,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    scaler = torch.cuda.amp.GradScaler() if DEVICE.type == 'cuda' else None

    BATCH_SIZE = 64
    EPOCHS = 30
    best_va = 0
    os.makedirs(MODEL_DIR, exist_ok=True)
    t_start = time.time()

    for ep in range(EPOCHS):
        # Train
        model.train()
        random.shuffle(train_data)
        total_loss, total_correct, total_n = 0.0, 0, 0

        for batch_start in range(0, len(train_data), BATCH_SIZE):
            batch = train_data[batch_start:batch_start + BATCH_SIZE]
            X, y = collate_fn(batch)
            X, y = X.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()

            if scaler:
                with torch.cuda.amp.autocast():
                    logits = model(X)
                    loss = criterion(logits, y)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                logits = model(X)
                loss = criterion(logits, y)
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * X.size(0)
            total_correct += (logits.argmax(1) == y).sum().item()
            total_n += X.size(0)

        tl = total_loss / total_n
        ta = total_correct / total_n

        # Val
        model.eval()
        total_loss, total_correct, total_n = 0.0, 0, 0
        with torch.no_grad():
            for batch_start in range(0, len(val_data), BATCH_SIZE):
                batch = val_data[batch_start:batch_start + BATCH_SIZE]
                X, y = collate_fn(batch)
                X, y = X.to(DEVICE), y.to(DEVICE)
                logits = model(X)
                loss = criterion(logits, y)
                total_loss += loss.item() * X.size(0)
                total_correct += (logits.argmax(1) == y).sum().item()
                total_n += X.size(0)

        vl = total_loss / total_n
        va = total_correct / total_n
        scheduler.step()

        if va > best_va:
            best_va = va
            torch.save(model.state_dict(), os.path.join(MODEL_DIR, 'best_model.pt'))

        elapsed = time.time() - t_start
        log_print(f"E{ep+1:3d}/{EPOCHS} | loss={tl:.4f} acc={ta:.4f} "
              f"vl={vl:.4f} va={va:.4f} | {elapsed:.0f}s")

    # 保存元数据
    pickle.dump({
        'c2i': c2i,
        'best_va': best_va,
        'n_speakers': n_speakers,
        'emb_dim': 256
    }, open(os.path.join(MODEL_DIR, 'xvector_torch_meta.pkl'), 'wb'))

    log_print(f"\n[DONE] Best VA: {best_va:.4f}, Total: {time.time()-t_start:.0f}s")


if __name__ == "__main__":
    train()
