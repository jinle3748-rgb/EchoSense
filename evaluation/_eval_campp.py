"""评测 CAM++ 在线追踪 25文件结果 vs 真实值"""
import json, os, sys
sys.path.insert(0, '.')
from rttm_parser import parse_rttm

with open('blind_online_25.json') as f:
    preds = json.load(f)

rttm_dir = r'c:\Users\hp\Desktop\SoundCUDA-main\labels\dev'
gt = {}
for fid in preds:
    rttm_path = os.path.join(rttm_dir, fid + '.rttm')
    if os.path.exists(rttm_path):
        segs = parse_rttm(rttm_path)
        speakers = set(s[2] for s in segs)
        gt[fid] = len(speakers)
    else:
        print(f'WARNING: no RTTM for {fid}')

print()
print('='*70)
print('CAM++ 在线追踪 vs 真实值 (25文件)')
print('='*70)
header = f'{"文件":12s}  {"预测":>6s}  {"真实":>6s}  {"误差":>6s}  {"正确?":>6s}'
print(header)
print('-'*50)

correct = 0
abs_error_sum = 0
for fid, pred_k in sorted(preds.items()):
    true_k = gt.get(fid, -1)
    if true_k == -1:
        print(f'{fid:12s}  {pred_k:>6d}  {"N/A":>6s}')
        continue
    error = pred_k - true_k
    abs_error_sum += abs(error)
    ok = 'YES' if abs(error) == 0 else '   NO'
    if abs(error) == 0:
        correct += 1
    print(f'{fid:12s}  {pred_k:>6d}  {true_k:>6d}  {error:>+6d}  {ok}')

n = len([f for f in preds if f in gt])
print('-'*50)
print(f'准确率 (Exact): {correct}/{n} = {correct/n*100:.1f}%')
print(f'MAE (平均绝对误差): {abs_error_sum/n:.2f}')
print(f'预测均值: {sum(preds.values())/len(preds):.2f}')
print(f'真实均值: {sum(gt[f] for f in preds if f in gt)/n:.2f}')
print()
print('错误详情:')
for fid, pred_k in sorted(preds.items()):
    true_k = gt.get(fid, -1)
    if true_k != -1 and abs(pred_k - true_k) > 0:
        print(f'  {fid}: 预测{pred_k} 真实{true_k} (差{pred_k-true_k:+d})')
