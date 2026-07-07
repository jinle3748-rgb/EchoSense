#!/usr/bin/env python3
"""对照 blind_216.json 和 RTTM 标签"""
import json, os, glob

PROJECT = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.dirname(os.path.dirname(PROJECT))
LABEL = os.path.join(BASE, 'labels', 'dev')

with open(os.path.join(PROJECT, 'blind_216.json')) as f:
    preds = json.load(f)

rttm_files = glob.glob(os.path.join(LABEL, '*.rttm'))
actual = {}
for rf in rttm_files:
    fid = os.path.splitext(os.path.basename(rf))[0]
    speakers = set()
    for line in open(rf):
        parts = line.strip().split()
        if len(parts) >= 8:
            speakers.add(parts[7])
    actual[fid] = len(speakers)

exact = 0
off1 = 0
bad = 0
by_actual = {}
for fid, pred_k in preds.items():
    ref_k = actual.get(fid, 0)
    diff = abs(pred_k - ref_k)
    if diff == 0:
        exact += 1
    if diff <= 1:
        off1 += 1
    if diff >= 3:
        bad += 1
    bucket = str(ref_k) if ref_k <= 5 else '6+'
    if bucket not in by_actual:
        by_actual[bucket] = {'total': 0, 'exact': 0, 'off1': 0}
    by_actual[bucket]['total'] += 1
    if diff == 0:
        by_actual[bucket]['exact'] += 1
    if diff <= 1:
        by_actual[bucket]['off1'] += 1

total = len(preds)
print(f"总计: {total} 条")
print(f"精确命中: {exact}/{total} ({100*exact/total:.1f}%)")
print(f"误差+/-1内: {off1}/{total} ({100*off1/total:.1f}%)")
print(f"严重偏差(>=3): {bad}/{total}")
print()
print(f"{'人数':>6}  {'文件数':>6}  {'命中率':>8}  {'+/-1率':>8}")
for k in sorted(by_actual.keys(), key=lambda x: 999 if x == '6+' else int(x)):
    d = by_actual[k]
    print(f"{k:>6}  {d['total']:>6}  {100*d['exact']/d['total']:>7.1f}%  {100*d['off1']/d['total']:>7.1f}%")
