"""
XVector训练 - PyTorch版
架构: TDNN(Conv1D) + StatisticsPooling + Embedding + Classifier
"""

import os, sys, time, pickle, glob, tarfile, warnings, json
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

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"[DEVICE] {DEVICE}")
if torch.cuda.is_available():
    print(f"[GPU] {torch.cuda.get_device_name(0)} | mem: {torch.cuda.get_device_properties(0).total_memory/1024**3:.1f}GB")
print("=" * 60, flush=True)

# =====================================================
# XVector Model (PyTorch)
# =====================================================
class TDNNBlock(nn.Module):
    """1D卷积 + BatchNorm + ReLU"""
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
        # x: (B, C, T) -> (B, 2*C)
        mean = x.mean(dim=2)
        std = x.std(dim=2)
        return torch.cat([mean, std], dim=1)

class XVector(nn.Module):
    def __init__(self, n_mfcc=20, n_classes=10, emb_dim=256):
        super().__init__()
        # Frame-level TDNN
        self.tdnn1 = TDNNBlock(n_mfcc, 512, 5)
        self.tdnn2 = TDNNBlock(512, 512, 3)
        self.tdnn3 = TDNNBlock(512, 512, 3)
        self.tdnn4 = TDNNBlock(512, 512, 1)
        self.tdnn5 = TDNNBlock(512, 1500, 1)

        # Pooling
        self.pool = StatisticsPooling()  # 1500 -> 3000

        # Segment-level
        self.fc1 = nn.Linear(3000, emb_dim)
        self.bn1 = nn.BatchNorm1d(emb_dim)
        self.fc2 = nn.Linear(emb_dim, n_classes)

        self.emb_dim = emb_dim
        self.dropout = nn.Dropout(0.3)

    def forward(self, x, return_emb=False):
        # x: (B, T, C) -> (B, C, T)
        x = x.transpose(1, 2)

        x = self.tdnn1(x)
        x = self.dropout(x)
        x = self.tdnn2(x)
        x = self.dropout(x)
        x = self.tdnn3(x)
        x = self.dropout(x)
        x = self.tdnn4(x)
        x = self.tdnn5(x)

        # Statistics pooling
        x = self.pool(x)
        x = x.view(x.size(0), -1)

        # Embedding
        emb = F.relu(self.bn1(self.fc1(x)))
        if return_emb:
            return emb

        # Classification
        return self.fc2(emb)


# =====================================================
# MFCC特征提取 (torchaudio)
# =====================================================
MFCC_TRANSFORM = torchaudio.transforms.MFCC(
    sample_rate=16000,
    n_mfcc=20,
    melkwargs={'n_fft': 512, 'hop_length': 160, 'n_mels': 20,
               'power': 2.0, 'window_fn': torch.hamming_window}
)

def extract_mfcc(waveform, sr=16000):
    """使用torchaudio提取MFCC"""
    if sr != 16000:
        resampler = torchaudio.transforms.Resample(sr, 16000)
        waveform = resampler(waveform)
    return MFCC_TRANSFORM(waveform)  # (1, n_mfcc, T)


# =====================================================
# 数据集
# =====================================================
class SpeakerDataset(Dataset):
    def __init__(self, audio_files, labels, seg_dur=2.0):
        self.segments = []
        self.targets = []
        seg_len = int(seg_dur * 16000)

        print(f"  加载 {len(audio_files)} 文件...", flush=True)
        t0 = time.time()

        # 建立标签映射
        classes = sorted(set(labels))
        self.c2i = {c: i for i, c in enumerate(classes)}

        loaded = 0
        for i, (path, lbl) in enumerate(zip(audio_files, labels)):
            if (i + 1) % 500 == 0:
                print(f"    进度: {i+1}/{len(audio_files)}", flush=True)
            try:
                if path.endswith('.wav'):
                    sr, y = wavfile.read(path)
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

                    y_tensor = torch.from_numpy(y).float().unsqueeze(0)

                    # 分段
                    n_seg = max(1, len(y) // seg_len)
                    for s in range(n_seg):
                        seg = y_tensor[:, s*seg_len:(s+1)*seg_len]
                        if seg.shape[1] < seg_len:
                            seg = F.pad(seg, (0, seg_len - seg.shape[1]))
                        # 提取MFCC
                        mfcc = extract_mfcc(seg)  # (1, 20, T_mfcc)
                        self.segments.append(mfcc.squeeze(0).transpose(0, 1))  # (T, 20)
                        self.targets.append(self.c2i[lbl])
                    loaded += 1
            except Exception as e:
                pass

        print(f"    加载: {loaded}文件, {len(self.segments)}片段 ({time.time()-t0:.1f}s)", flush=True)

    def __len__(self):
        return len(self.segments)

    def __getitem__(self, idx):
        return self.segments[idx], self.targets[idx]


def collate_fn(batch):
    """填充到相同长度"""
    features, labels = zip(*batch)
    max_len = max(f.shape[0] for f in features)
    padded = []
    for f in features:
        if f.shape[0] < max_len:
            pad = torch.zeros(max_len - f.shape[0], f.shape[1])
            f = torch.cat([f, pad], dim=0)
        padded.append(f)
    return torch.stack(padded), torch.tensor(labels)


# =====================================================
# 训练
# =====================================================
def train_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss, total_correct, total_n = 0.0, 0, 0
    for X, y in loader:
        X, y = X.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        logits = model(X)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * X.size(0)
        total_correct += (logits.argmax(1) == y).sum().item()
        total_n += X.size(0)
    return total_loss / total_n, total_correct / total_n


@torch.no_grad()
def eval_epoch(model, loader, criterion):
    model.eval()
    total_loss, total_correct, total_n = 0.0, 0, 0
    for X, y in loader:
        X, y = X.to(DEVICE), y.to(DEVICE)
        logits = model(X)
        loss = criterion(logits, y)
        total_loss += loss.item() * X.size(0)
        total_correct += (logits.argmax(1) == y).sum().item()
        total_n += X.size(0)
    return total_loss / total_n, total_correct / total_n


def main():
    BASE = os.path.dirname(os.path.abspath(__file__))
    DATA = os.path.join(BASE, "RecogizeTrain/data/raw/data_aishell")
    MODEL_DIR = os.path.join(BASE, "models/XVector_full")
    SEG_DUR = 2.0  # 2秒片段
    BATCH_SIZE = 32
    EPOCHS = 50
    LR = 0.001

    # 解压
    td = os.path.join(DATA, "wav", "train")
    for tf in glob.glob(os.path.join(td, "*.tar.gz")):
        sid = os.path.basename(tf).replace('.tar.gz', '')
        if not os.path.exists(os.path.join(td, sid)):
            tarfile.open(tf, 'r:gz').extractall(td)

    # 收集数据
    af, lb = [], []
    for d in sorted(os.listdir(td)):
        dp = os.path.join(td, d)
        if os.path.isdir(dp):
            for w in glob.glob(os.path.join(dp, "*.wav")):
                af.append(w); lb.append(d)

    n_speakers = len(set(lb))
    print(f"说话人: {n_speakers} | 音频: {len(af)}", flush=True)

    # 使用全部数据（不再限制3000文件）
    n_speakers = len(set(lb))
    print(f"[训练集] {len(af)}文件, {n_speakers}说话人", flush=True)

    # 划分
    np.random.seed(42)
    idx = np.random.permutation(len(af))
    sp = int(len(af) * 0.8)
    train_files = [af[i] for i in idx[:sp]]
    train_labels = [lb[i] for i in idx[:sp]]
    val_files = [af[i] for i in idx[sp:]]
    val_labels = [lb[i] for i in idx[sp:]]

    # 数据集
    print("\n[训练数据]", flush=True)
    train_ds = SpeakerDataset(train_files, train_labels, seg_dur=SEG_DUR)
    print("[验证数据]", flush=True)
    val_ds = SpeakerDataset(val_files, val_labels, seg_dur=SEG_DUR)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              collate_fn=collate_fn, num_workers=0, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                            collate_fn=collate_fn, num_workers=0, pin_memory=True)

    print(f"\n[数据] 训练: {len(train_ds)} 验证: {len(val_ds)}", flush=True)

    # 模型
    model = XVector(n_mfcc=20, n_classes=n_speakers, emb_dim=256).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[模型] 参数: {n_params:,}", flush=True)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.5)

    best_va = 0
    history = {'loss': [], 'acc': [], 'vloss': [], 'vacc': []}
    t0 = time.time()

    for ep in range(EPOCHS):
        tl, ta = train_epoch(model, train_loader, optimizer, criterion)
        vl, va = eval_epoch(model, val_loader, criterion)
        scheduler.step()

        history['loss'].append(tl)
        history['acc'].append(ta)
        history['vloss'].append(vl)
        history['vacc'].append(va)

        if va > best_va:
            best_va = va
            torch.save(model.state_dict(), os.path.join(MODEL_DIR, 'best_model.pt'))

        if (ep + 1) % 5 == 0 or ep == 0:
            print(f"  E{ep+1:3d}/{EPOCHS} | loss={tl:.4f} acc={ta:.4f} "
                  f"vl={vl:.4f} va={va:.4f} | {time.time()-t0:.1f}s", flush=True)

    print(f"\n完成! 最佳VA: {best_va:.4f} 耗时: {time.time()-t0:.1f}s", flush=True)

    # 保存
    os.makedirs(MODEL_DIR, exist_ok=True)
    pickle.dump({
        'c2i': train_ds.c2i,
        'history': history,
        'best_va': best_va,
        'n_speakers': n_speakers,
        'emb_dim': 256
    }, open(os.path.join(MODEL_DIR, 'xvector_torch_meta.pkl'), 'wb'))
    print(f"模型已保存: {MODEL_DIR}", flush=True)


if __name__ == "__main__":
    main()
