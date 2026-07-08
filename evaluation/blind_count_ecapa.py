#!/usr/bin/env python3
"""
纯盲说话人计数 — 在线追踪法 (ECAPA-TDNN 嵌入)
与 blind_count.py 共用相同的算法逻辑，仅嵌入模型不同
"""
import os, sys, time, warnings
warnings.filterwarnings('ignore')

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE = os.path.dirname(PROJECT)
sys.path.insert(0, PROJECT)

import numpy as np
import torch
import json
import shutil

_t_load = torch.load
torch.load = lambda *a, **kw: _t_load(*a, **{**kw, 'weights_only': False})

import pathlib
if hasattr(pathlib.WindowsPath, 'symlink_to'):
    _orig_symlink = pathlib.WindowsPath.symlink_to
    def _copy_instead_of_symlink(self, target, target_is_directory=False):
        target_path = pathlib.Path(target)
        if target_path.is_file():
            shutil.copy2(str(target_path), str(self))
        elif target_path.is_dir():
            shutil.copytree(str(target_path), str(self), dirs_exist_ok=True)
    pathlib.WindowsPath.symlink_to = _copy_instead_of_symlink

# ---- 导入共享的 VAD / 分割 / 在线追踪函数 ----
from blind_count import (
    AUDIO, SR_TARGET, TEST_25,
    energy_vad, merge_short_segments, extract_timeline_embeddings,
    online_track, count_speakers
)

# ---- ECAPA-TDNN 参数 (SpeechBrain) ----
THRESHOLD_MATCH = 0.52      # ECAPA 192维最佳阈值
EMA_ALPHA = 0.35
MIN_CONFIRM = 3
MIN_SEG_DUR = 3.5
CHUNK_DUR = 3.0

# ---- ECAPA-TDNN 嵌入 (SpeechBrain VoxCeleb) ----
_embedding_model = None
_EMB_SIZE = 192

def get_embedding_model():
    global _embedding_model
    if _embedding_model is None:
        from speechbrain.inference.speaker import EncoderClassifier
        print("[Embedding] 加载 ECAPA-TDNN (192维, SpeechBrain)...")
        _embedding_model = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir=os.path.join(BASE, 'pretrained_models', 'ecapa'),
            run_opts={"device": "cuda:0" if torch.cuda.is_available() else "cpu"}
        )
        print(f"[Embedding] 加载完成")
    return _embedding_model


def extract_one_embedding(chunk, sr, model):
    """从一段音频提取 ECAPA 嵌入 (192-dim)"""
    # SpeechBrain 期望 float32 tensor
    chunk_t = torch.from_numpy(chunk).float().unsqueeze(0)
    device = next(model.parameters()).device if hasattr(next(model.parameters()), 'device') else 'cpu'
    chunk_t = chunk_t.to(device)
    with torch.no_grad():
        emb = model.encode_batch(chunk_t)
    return emb.squeeze().cpu().numpy()


# ---- 覆盖模块级的 embedding 函数 ----
import blind_count as _bc
_bc.THRESHOLD_MATCH = THRESHOLD_MATCH
_bc.EMA_ALPHA = EMA_ALPHA
_bc.MIN_CONFIRM = MIN_CONFIRM
_bc.MIN_SEG_DUR = MIN_SEG_DUR
_bc.CHUNK_DUR = CHUNK_DUR
_bc.get_embedding_model = get_embedding_model
_bc.extract_one_embedding = extract_one_embedding


# ---- 主入口 ----
MODE_25_ONLY = False  # True=只测25条  False=全216条

def main():
    import glob
    if MODE_25_ONLY:
        wav_files = [os.path.join(AUDIO, f + '.wav') for f in TEST_25
                     if os.path.exists(os.path.join(AUDIO, f + '.wav'))]
        out_name = 'blind_ecapa_25.json'
    else:
        wav_files = sorted(glob.glob(os.path.join(AUDIO, '*.wav')))
        out_name = 'blind_ecapa_216.json'
    total = len(wav_files)

    print("=" * 60)
    print(f"  纯盲说话人计数 — 在线追踪法 ECAPA ({total}条)")
    print(f"  阈值: match>{THRESHOLD_MATCH:.2f}  min确认={MIN_CONFIRM}  EMA={EMA_ALPHA}")
    print("=" * 60)

    model = get_embedding_model()
    predictions = {}
    out_path = os.path.join(PROJECT, 'evaluation', out_name)

    for i, wav in enumerate(wav_files):
        fid = os.path.splitext(os.path.basename(wav))[0]
        t0 = time.time()
        k, dur, n_segs, trace = count_speakers(wav, model, verbose=False)
        elapsed = time.time() - t0

        n_profiles = len([l for l in trace if '档案' in l and '↑' not in l])
        n_cands = len([l for l in trace if '新候选' in l])
        print(f"  [{i+1:>3}/{total}] {fid}  dur={dur:5.0f}s  "
              f"预测={k:>2}人  segs={n_segs:>3}  档案={n_profiles}  候选={n_cands}  {elapsed:4.0f}s")
        predictions[fid] = k

        if (i + 1) % 10 == 0:
            with open(out_path, 'w') as f:
                json.dump(predictions, f, indent=2, ensure_ascii=False)

    with open(out_path, 'w') as f:
        json.dump(predictions, f, indent=2, ensure_ascii=False)
    print(f"\n预测结果已保存到 evaluation/{out_name}")


if __name__ == '__main__':
    main()
