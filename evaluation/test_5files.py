"""小批量测试：5个文件验证 ECAPA 评估管线"""
import os, sys, time, numpy as np

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PARENT_DIR = os.path.dirname(PROJECT_DIR)
sys.path.insert(0, PROJECT_DIR)
sys.path.insert(0, os.path.join(PROJECT_DIR, 'ECAPA-TDNN'))

from evaluation.rttm_parser import rttm_to_frame_labels
from evaluation.diarization_eval import evaluate_one
from ecapa_speaker import analyze_speakers
import librosa

# 输出同时写文件
log_path = os.path.join(PROJECT_DIR, 'evaluation', '_test_output.txt')
log = open(log_path, 'w', encoding='utf-8')

def log_print(msg):
    print(msg, flush=True)
    log.write(msg + '\n')
    log.flush()

RTTM_DIR = os.path.join(PARENT_DIR, 'labels', 'dev')
AUDIO_DIR = os.path.join(PARENT_DIR, 'voxconverse_dev_wav', 'audio')
SEG_DUR, OVERLAP = 1.0, 0.5
HOP = SEG_DUR * (1 - OVERLAP)

test_files = ['abjxc', 'bdopb', 'crixb', 'dgxgn', 'eapdk']
results = []

for fid in test_files:
    t0 = time.time()
    audio_path = os.path.join(AUDIO_DIR, f'{fid}.wav')
    rttm_path = os.path.join(RTTM_DIR, f'{fid}.rttm')

    dur = librosa.get_duration(filename=audio_path)
    n_frames = max(1, int(dur / HOP) + 1)
    ref_labels, ref_n_spk = rttm_to_frame_labels(rttm_path, SEG_DUR, OVERLAP, dur)

    result = analyze_speakers(audio_path, seg_dur=SEG_DUR, overlap=OVERLAP)
    hyp_labels = [-1] * n_frames
    for i, (start, spk_id) in enumerate(result['timeline']):
        if i < n_frames and spk_id >= 0:
            hyp_labels[i] = int(spk_id)

    ev = evaluate_one(ref_labels, hyp_labels, ref_n_spk)
    elapsed = time.time() - t0
    hyp_n = len(set(h for h in hyp_labels if h >= 0))
    log_print(f"{fid}: DER={ev['DER']:.1f}%, 帧acc={ev['frame_accuracy']:.1f}%, "
              f"参考{ref_n_spk}人/预测{hyp_n}人, {elapsed:.1f}s")
    results.append(ev)

ders = [r['DER'] for r in results]
accs = [r['frame_accuracy'] for r in results]
log_print(f"\n==== 汇总 ====")
log_print(f"成功: {len(results)}/5")
log_print(f"DER 均值: {np.mean(ders):.2f}%")
log_print(f"帧准确率均值: {np.mean(accs):.2f}%")
log.close()
