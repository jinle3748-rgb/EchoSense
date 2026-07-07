#!/usr/bin/env python3
"""诊断：打印每个文件 k=1..6 的 BIC/AIC 值，优化参数"""
import os, sys, json
sys.path.insert(0, '.')
sys.path.insert(0, 'ECAPA-TDNN')

import librosa, numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from ecapa_speaker import extract_embeddings

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE = os.path.dirname(PROJECT)  # 上一层才是 voxconverse_dev_wav/labels 所在
AUDIO = os.path.join(BASE, 'voxconverse_dev_wav', 'audio')
LABELS = os.path.join(BASE, 'labels', 'dev')
RESULTS = os.path.join(PROJECT, 'evaluation', 'medium_test_results.json')

with open(RESULTS) as f:
    data = json.load(f)

# 只分析预测 k=1 但真实是 2+ 的文件
focus = [d for d in data if d['ref_n_speakers'] >= 2 and d['hyp_n_speakers'] == 1]

print("=" * 75)
print("  2+ 说话人被判为 1 人的文件 — BIC/AIC 分析")
print("=" * 75)

for d in focus:
    fid = d['file_id']
    wav = os.path.join(AUDIO, f'{fid}.wav')
    dur = d['duration']

    embeddings, _, _ = extract_embeddings(wav, seg_dur=1.0, overlap=0.5)
    scaler = StandardScaler()
    emb_scaled = scaler.fit_transform(embeddings)

    n = embeddings.shape[0]
    pca_dim = min(32, n - 2, emb_scaled.shape[1])
    pca = PCA(n_components=pca_dim) if pca_dim >= 2 else None
    emb = pca.fit_transform(emb_scaled) if pca else emb_scaled[:, :2]

    rows = []
    for k in range(1, 7):
        if k > n // 2:
            break
        gmm = GaussianMixture(n_components=k, covariance_type='diag',
                              n_init=3, max_iter=200, random_state=42)
        gmm.fit(emb)
        bic = gmm.bic(emb)
        aic = gmm.aic(emb)
        rows.append((k, bic, aic))

    print(f"\n--- {fid} (ref={d['ref_n_speakers']}人, dur={dur}s, segs={n}) ---")
    print(f"{'k':>3} {'BIC':>10} {'AIC':>10} {'ΔBIC(k-1)':>12}")
    for i, (k, bic, aic) in enumerate(rows):
        delta = "" if i == 0 else f"{bic - rows[0][1]:>+12.1f}"
        print(f"{k:>3} {bic:>10.1f} {aic:>10.1f} {delta}")

print("\n" + "=" * 75)
print("结论: 看 ΔBIC(k=2 - k=1)。如果这个值很小 (<5)，说明 BIC 在 k=1/2 之间难以区分")
print("      需要调高阈值，或者改用 AIC delta。")
