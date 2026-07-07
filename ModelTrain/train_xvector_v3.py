"""XVector V3 - Train from preloaded pickle data (fast!)"""
import os, sys, pickle, time, random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, 'models', 'XVector_full')
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"[DEVICE] {DEVICE}", flush=True)

# Import model
sys.path.insert(0, BASE_DIR)
from train_xvector_v2 import XVector, collate_fn

# Load preloaded data
pkl_path = os.path.join(MODEL_DIR, 'preloaded_data.pkl')
print(f"[加载] {pkl_path}...", flush=True)
data = pickle.load(open(pkl_path, 'rb'))
train_data = data['train']
val_data = data['val']
c2i = data['c2i']
n_speakers = data['n_speakers']
print(f"[数据] 训练: {len(train_data)}, 验证: {len(val_data)}, 说话人: {n_speakers}", flush=True)

# Model
model = XVector(n_mfcc=20, n_classes=n_speakers, emb_dim=256).to(DEVICE)
n_params = sum(p.numel() for p in model.parameters())
print(f"[模型] 参数: {n_params:,}", flush=True)

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
    print(f"E{ep+1:3d}/{EPOCHS} | loss={tl:.4f} acc={ta:.4f} "
          f"vl={vl:.4f} va={va:.4f} | {elapsed:.0f}s", flush=True)

# Save metadata
pickle.dump({
    'c2i': c2i,
    'best_va': best_va,
    'n_speakers': n_speakers,
    'emb_dim': 256
}, open(os.path.join(MODEL_DIR, 'xvector_torch_meta.pkl'), 'wb'))

print(f"\n[DONE] Best VA: {best_va:.4f}, Total: {time.time()-t_start:.0f}s", flush=True)
