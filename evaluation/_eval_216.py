"""评测全216文件结果"""
import json, os, sys
sys.path.insert(0, '.')
from rttm_parser import parse_rttm

with open('blind_online_track.json') as f:
    preds = json.load(f)

rttm_dir = r'c:\Users\hp\Desktop\SoundCUDA-main\labels\dev'
gt = {}
missing = []
for fid in preds:
    rttm_path = os.path.join(rttm_dir, fid + '.rttm')
    if os.path.exists(rttm_path):
        segs = parse_rttm(rttm_path)
        speakers = set(s[2] for s in segs)
        gt[fid] = len(speakers)
    else:
        missing.append(fid)

n = len(gt)
correct = sum(1 for f in gt if preds[f] == gt[f])
abs_err_sum = sum(abs(preds[f] - gt[f]) for f in gt)
mae = abs_err_sum / n

print('='*60)
print(f'CAM++ 在线追踪  全{len(preds)}文件 (含RTTM: {n})')
print(f'参数: th=0.55 ema=0.25 mc=4')
print('='*60)
print(f'准确率 (Exact): {correct}/{n} = {correct/n*100:.1f}%')
print(f'MAE: {mae:.3f}')
print(f'预测均值: {sum(preds.values())/len(preds):.2f}')
print(f'真实均值: {sum(gt.values())/n:.2f}')

# 误差分布
errors = [preds[f] - gt[f] for f in gt]
over  = sum(1 for e in errors if e > 0)
under = sum(1 for e in errors if e < 0)
print(f'\n误差分布: 高估={over}  低估={under}  正确={correct}')
print(f'最大高估: +{max(errors)}  最大低估: {min(errors)}')

# 误差±1以内
within1 = sum(1 for e in errors if abs(e) <= 1)
print(f'误差≤1: {within1}/{n} = {within1/n*100:.1f}%')

# 按真实人数分组统计
print(f'\n按真实人数分组:')
from collections import defaultdict
by_true = defaultdict(list)
for f in gt:
    by_true[gt[f]].append(preds[f])
for k in sorted(by_true):
    vals = by_true[k]
    c = sum(1 for v in vals if v == k)
    print(f'  真{k}人 ({len(vals)}文件): 准确{c}个={c/len(vals)*100:.0f}%  预测均值={sum(vals)/len(vals):.1f}')

# 错误最多的
print(f'\n错误最严重的10个:')
worst = sorted([(f, abs(preds[f]-gt[f]), preds[f], gt[f]) for f in gt if preds[f] != gt[f]],
               key=lambda x: -x[1])
for fid, diff, pred, true in worst[:10]:
    print(f'  {fid}: 预测{pred} 真实{true} (差{pred-true:+d})')

if missing:
    print(f'\n缺失RTTM: {len(missing)}个 ({missing[:5]}...)')
