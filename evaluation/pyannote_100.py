#!/usr/bin/env python3
"""pyannote 100文件测试 — 验证1-4人场景"""
import os, sys, time

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE = os.path.dirname(PROJECT)
sys.path.insert(0, PROJECT)
sys.path.insert(0, os.path.join(PROJECT, 'evaluation'))

import numpy as np, json, random, gc
import soundfile as sf
import torch
from pyannote.audio import Pipeline
from rttm_parser import rttm_to_frame_labels
from diarization_eval import evaluate_one

AUDIO = os.path.join(BASE, 'voxconverse_dev_wav', 'audio')
LABELS = os.path.join(BASE, 'labels', 'dev')
SEG_DUR, OVERLAP = 0.5, 0.25

# 从125个1-4人文件中随机抽取100个（seed=42）
random.seed(42)
TEST_FILES = ['qpylu','cmhsm','ampme','syiwe','hiyis','gqbvk','fxgvy','cwryz','zrlyl',
              'bydui','qygfk','yuzyu','oenox','bspxd','pqmho','kkghn','asxwr','ztzzr',
              'ycxxe','femmv','gofnj','mvjuk','qfdpp','wmori','paibn','evtyi','sldwj',
              'qsfzo','rtvuw','yfcmz','kctgl','zajzs','mekog','xypdm','zmndm','abjxc',
              'djngn','xvllq','iqbww','sikkm','dhorc','whmpa','yrsve','ysgbf','willh',
              'jiqvr','bwzyf','irvat','iqtde','grzbb','atgpi','mesob','nxgad','cobal',
              'qouur','bravd','houcx','iwdjy','eqttu','bkwns','ppgjx','vysqj','jnivh',
              'cyyxp','vmbga','qzwxa','zvmyn','kklpv','aufkn','ngyrk','ywcwr','mpvoh',
              'ikgcq','ehpau','szsyz','jyirt','dscgs','zyffh','ioasm','mevkw','praxo',
              'qrzjk','imtug','xiglo','hqyok','jsdmu','gwtwd','uatlu','hycgx','ypwjd',
              'sosnj','uvnmy','plbbw','rcxzg','ezsgk','kckqn','qydmg','ahnss','azisu',
              'tfvyr']

print("=" * 65)
print("  pyannote 100 文件测试 — 验证 1-4 人场景")
print("=" * 65)

print("\n[pyannote] 加载模型...")
pipeline = Pipeline.from_pretrained('pyannote/speaker-diarization-3.1',
    token='YOUR_HF_TOKEN')  # 请替换为你的 HuggingFace token
pipeline.to(torch.device('cpu'))
print("[pyannote] 加载完成\n")

results = []
tot_elapsed = 0

# 断点续跑：读取已有结果
out_path = os.path.join(PROJECT, 'evaluation', 'pyannote_100_results.json')
done_files = set()
if os.path.exists(out_path):
    with open(out_path) as f:
        results = json.load(f)
    done_files = {r['file_id'] for r in results}
    print(f"[续跑] 已有 {len(results)} 条结果，跳过已完成文件\n")
for i, fid in enumerate(TEST_FILES):
    wav = os.path.join(AUDIO, f'{fid}.wav')
    rttm = os.path.join(LABELS, f'{fid}.rttm')
    if not os.path.exists(wav):
        print(f"  [{i+1:>3}/100] {fid}: 跳过（文件不存在）")
        continue
    if fid in done_files:
        continue

    y, sr = sf.read(wav, dtype='float32')
    if y.ndim > 1:
        y = y.mean(axis=1)
    dur = len(y) / sr
    n_frames = int(dur / (SEG_DUR * (1 - OVERLAP))) + 1

    ref_labels, ref_n = rttm_to_frame_labels(rttm, SEG_DUR, OVERLAP, dur)

    audio_dict = {'waveform': torch.from_numpy(y).unsqueeze(0), 'sample_rate': sr}

    t0 = time.time()
    dia = pipeline(audio_dict)
    elapsed = time.time() - t0
    tot_elapsed += elapsed

    annotation = dia.speaker_diarization
    hyp_labels = np.full(n_frames, -1, dtype=int)
    spk_map = {}
    next_spk = 0
    for turn, _, speaker in annotation.itertracks(yield_label=True):
        fs = int(turn.start / (SEG_DUR * (1 - OVERLAP)))
        fe = int(turn.end / (SEG_DUR * (1 - OVERLAP))) + 1
        if speaker not in spk_map:
            spk_map[speaker] = next_spk
            next_spk += 1
        hyp_labels[max(0, fs):min(n_frames, fe)] = spk_map[speaker]

    ev = evaluate_one(ref_labels, hyp_labels, ref_n)

    print(f"  [{i+1:>3}/100] {fid}  dur={dur:5.0f}s  ref={ref_n}  pred={next_spk}"
          f"  DER={ev['DER']:5.1f}%  acc={ev['frame_accuracy']:5.1f}%  "
          f"spk_ok={ev['n_spk_correct']}  {elapsed:4.0f}s")
    results.append({'file_id': fid, 'duration': dur, 'ref_n_speakers': ref_n,
                    'pred_n_speakers': next_spk, 'n_spk_correct': ev['n_spk_correct'],
                    'DER': ev['DER'], 'frame_accuracy': ev['frame_accuracy'],
                    'elapsed_s': elapsed, 'model': 'pyannote'})

    # 每10个文件增量保存，防止崩溃丢失
    if (i + 1) % 10 == 0:
        with open(os.path.join(PROJECT, 'evaluation', 'pyannote_100_results.json'), 'w') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"  --- 已保存 {i+1} 条结果 ---")

    # 释放内存
    del y, audio_dict, dia, annotation, hyp_labels, ref_labels
    gc.collect()

# 按说话人数分组统计
buckets = {1: [], 2: [], 3: [], 4: []}
for r in results:
    k = r['ref_n_speakers']
    if k in buckets:
        buckets[k].append(r)

print("\n" + "=" * 65)
print(f"  {'场景':<10} {'文件数':<6} {'人数正确':<10} {'平均 DER':<10} {'平均 acc':<10}")
print("-" * 65)
total_ok = 0
for k in sorted(buckets):
    b = buckets[k]
    if not b:
        continue
    n_ok = sum(1 for r in b if r['n_spk_correct'])
    avg_der = sum(r['DER'] for r in b) / len(b)
    avg_acc = sum(r['frame_accuracy'] for r in b) / len(b)
    total_ok += n_ok
    print(f"  {f'{k}人':<10} {len(b):<6} {f'{n_ok}/{len(b)}':<10} {avg_der:<10.1f} {avg_acc:<10.1f}")
print("-" * 65)
print(f"  {'合计':<10} {len(results):<6} {f'{total_ok}/{len(results)}':<10}")
print(f"  总耗时: {tot_elapsed:.0f}s ({tot_elapsed/60:.1f}min)")
print("=" * 65)

# 保存
out_dir = os.path.join(PROJECT, 'evaluation')
with open(os.path.join(out_dir, 'pyannote_100_results.json'), 'w') as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
print(f"\n结果已保存到 evaluation/pyannote_100_results.json")
