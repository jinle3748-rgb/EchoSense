"""评测 ECAPA 全216文件结果"""
import json, os, sys
sys.path.insert(0, '.')
from rttm_parser import parse_rttm

with open('blind_ecapa_216.json') as f:
    preds = json.load(f)

rttm_dir = r'c:\Users\hp\Desktop\SoundCUDA-main\labels\dev'
gt = {}
for fid in preds:
    rttm_path = os.path.join(rttm_dir, fid + '.rttm')
    if os.path.exists(rttm_path):
        segs = parse_rttm(rttm_path)
        gt[fid] = len(set(s[2] for s in segs))

n = len(gt)
correct = sum(1 for f in gt if preds[f] == gt[f])
abs_err_sum = sum(abs(preds[f] - gt[f]) for f in gt)
mae = abs_err_sum / n
errors = [preds[f] - gt[f] for f in gt]
within1 = sum(1 for e in errors if abs(e) <= 1)

print('='*60)
print(f'ECAPA 在线追踪  全{len(preds)}文件 (含RTTM: {n})')
print(f'参数: th=0.52 ema=0.35 mc=3')
print('='*60)
print(f'准确率 (Exact): {correct}/{n} = {correct/n*100:.1f}%')
print(f'MAE: {mae:.3f}')
print(f'误差≤1: {within1}/{n} = {within1/n*100:.1f}%')

from collections import defaultdict
by_true = defaultdict(list)
for f in gt:
    by_true[gt[f]].append(preds[f])
print(f'\n按真实人数分组:')
for k in sorted(by_true):
    vals = by_true[k]
    c = sum(1 for v in vals if v == k)
    print(f'  真{k}人 ({len(vals)}文件): 准确{c}个={c/len(vals)*100:.0f}%  预测均值={sum(vals)/len(vals):.1f}')
