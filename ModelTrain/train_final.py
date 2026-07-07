"""
XVector Training - 25 speakers, memory-loaded, self-contained
Saves bundled .pt for future fast loading
"""
import os, sys, time, glob, re, random, gc
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pickle

# === Config ===
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, 'models', 'XVector_full')
CACHE_DIR = os.path.join(MODEL_DIR, 'mfcc_cache')
BUNDLE_FILE = os.path.join(MODEL_DIR, 'bundle_25spk.pt')
LOG_FILE = os.path.join(os.path.dirname(BASE_DIR), 'train_log.txt')
N_SPEAKERS = 25
BATCH_SIZE = 64
EPOCHS = 30

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

_fh = open(LOG_FILE, 'a', encoding='utf-8')
def log(msg):
    _fh.write(msg + '\n'); _fh.flush()
    try: print(msg, flush=True)
    except: pass

log(f"=== XVector Training: {N_SPEAKERS} speakers ===")
log(f"[DEVICE] {DEVICE}")
if DEVICE.type == 'cuda': log(f"[GPU] {torch.cuda.get_device_name(0)}")

# === Model ===
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
    def forward(self, x):
        emb = self.t(x.transpose(1,2))
        return self.cls(emb)
    def get_embedding(self, x):
        return self.t(x.transpose(1,2))

# =============================================================
# Phase 1: Load or Build Data Bundle
# =============================================================
t0 = time.time()
log(f"[Phase 1] Data preparation...")

if os.path.exists(BUNDLE_FILE):
    log(f"  Loading bundle: {BUNDLE_FILE}")
    bundle = torch.load(BUNDLE_FILE, map_location='cpu')
    train_data, val_data = bundle['train'], bundle['val']
    s2i = bundle['s2i']
    log(f"  Loaded! Train: {len(train_data)}, Val: {len(val_data)}, Speakers: {len(s2i)} ({time.time()-t0:.0f}s)")
else:
    log("  Scanning .npy files...")
    npy_files = glob.glob(os.path.join(CACHE_DIR, '*.npy'))
    log(f"  Found {len(npy_files)} total files")

    # Map files to speakers
    sk = re.compile(r'(S\d+)')
    file_spk = {}
    for f in npy_files:
        m = sk.search(os.path.basename(f))
        if m: file_spk[f] = m.group(1)

    all_spks = sorted(set(file_spk.values()))
    random.seed(42)
    selected = sorted(random.sample(all_spks, N_SPEAKERS))
    log(f"  Selected {len(selected)} speakers: {selected}")

    # Load all matching files
    log("  Loading .npy files to memory...")
    data = []  # (tensor, label)
    skipped = 0
    t_load = time.time()
    for i, (f, spk) in enumerate(file_spk.items()):
        if spk not in selected: continue
        if (len(data) + 1) % 2000 == 0:
            elapsed = time.time() - t_load
            log(f"    {len(data)+1} loaded ({elapsed:.0f}s, {skipped} skipped)")
        try:
            data.append((torch.from_numpy(np.load(f)).float(), spk))
        except Exception:
            skipped += 1

    load_time = time.time() - t_load
    log(f"  Loaded {len(data)} files, {skipped} skipped ({load_time:.0f}s)")

    # Shuffle and split
    np.random.seed(42)
    idx = np.random.permutation(len(data))
    sp = int(len(data) * 0.8)

    # Encode labels
    spk_set = sorted(set(spk for _, spk in data))
    s2i = {s: i for i, s in enumerate(spk_set)}
    train_data = [(t, s2i[spk]) for t, spk in [data[i] for i in idx[:sp]]]
    val_data = [(t, s2i[spk]) for t, spk in [data[i] for i in idx[sp:]]]
    log(f"  Split: Train={len(train_data)}, Val={len(val_data)}, Speakers={len(s2i)}")

    # Save bundle for future fast loading
    log(f"  Saving bundle...")
    bundle = {'train': train_data, 'val': val_data, 's2i': s2i}
    torch.save(bundle, BUNDLE_FILE)
    log(f"  Saved: {os.path.getsize(BUNDLE_FILE)/(1024*1024):.1f}MB")

del bundle; gc.collect()
log(f"[Phase 1] Done ({time.time()-t0:.0f}s)")

# =============================================================
# Phase 2: Training
# =============================================================
n_speakers = len(s2i)
c2i = s2i

def collate_fn(batch):
    tensors, labels = zip(*batch)
    mx = max(t.size(0) for t in tensors)
    pad = torch.zeros(len(tensors), mx, tensors[0].size(1))
    for i, t in enumerate(tensors): pad[i, :t.size(0), :] = t
    return pad, torch.tensor(labels)

def batch_iterator(data, batch_size, shuffle):
    idx = list(range(len(data)))
    if shuffle: np.random.shuffle(idx)
    for i in range(0, len(idx) - batch_size + 1, batch_size):
        batch_idx = idx[i:i + batch_size]
        yield collate_fn([data[j] for j in batch_idx])

n_train_batches = len(train_data) // BATCH_SIZE
n_val_batches = len(val_data) // BATCH_SIZE
log(f"[Phase 2] Training: {n_train_batches} batches/epoch, {EPOCHS} epochs")

model = XVector(20, n_speakers, 256).to(DEVICE)
log(f"  Model params: {sum(p.numel() for p in model.parameters()):,}")

crit = nn.CrossEntropyLoss()
opt = optim.Adam(model.parameters(), lr=0.001)
sch = optim.lr_scheduler.StepLR(opt, 10, 0.5)
sc = torch.amp.GradScaler('cuda') if DEVICE.type == 'cuda' else None
best_va = 0.0
t_train = time.time()

for ep in range(EPOCHS):
    # Train
    model.train(); tl, tc = 0.0, 0
    for X, y in batch_iterator(train_data, BATCH_SIZE, shuffle=True):
        X, y = X.to(DEVICE), y.to(DEVICE); opt.zero_grad()
        if sc:
            with torch.amp.autocast('cuda'): lo = model(X); ls = crit(lo, y)
            sc.scale(ls).backward(); sc.step(opt); sc.update()
        else:
            lo = model(X); ls = crit(lo, y); ls.backward(); opt.step()
        tl += ls.item() * X.size(0); tc += (lo.argmax(1) == y).sum().item()

    ta = tc / (n_train_batches * BATCH_SIZE)
    t_loss = tl / (n_train_batches * BATCH_SIZE)

    # Validate
    model.eval(); vl, vc = 0.0, 0
    with torch.no_grad():
        for X, y in batch_iterator(val_data, BATCH_SIZE, shuffle=False):
            X, y = X.to(DEVICE), y.to(DEVICE)
            lo = model(X); ls = crit(lo, y)
            vl += ls.item() * X.size(0); vc += (lo.argmax(1) == y).sum().item()
    va = vc / (n_val_batches * BATCH_SIZE)
    v_loss = vl / (n_val_batches * BATCH_SIZE)
    sch.step()

    if va > best_va:
        best_va = va
        torch.save(model.state_dict(), os.path.join(MODEL_DIR, 'best_model.pt'))

    elapsed = time.time() - t_train
    log(f"E{ep+1:3d}/{EPOCHS} | L={t_loss:.4f} A={ta:.4f} | VL={v_loss:.4f} VA={va:.4f} | {elapsed:.0f}s | LR={opt.param_groups[0]['lr']:.6f}")

# =============================================================
# Save meta
# =============================================================
pickle.dump({
    'c2i': c2i, 'best_va': best_va,
    'n_speakers': n_speakers, 'emb_dim': 256
}, open(os.path.join(MODEL_DIR, 'xvector_torch_meta.pkl'), 'wb'))

log(f"[DONE] BestVA={best_va:.4f} | Total={time.time()-t0:.0f}s | Speakers={n_speakers}")
_fh.close()
