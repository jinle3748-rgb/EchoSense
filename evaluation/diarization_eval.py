"""
核心评估逻辑 — 比较模型输出与 RTTM 标注
"""
import numpy as np
from scipy.optimize import linear_sum_assignment
from collections import Counter


def compute_confusion_matrix(ref_labels, hyp_labels):
    """
    构建参考 vs 预测的混淆矩阵（仅考虑有声部分）。

    返回:
        confusion:  numpy array (n_ref_spk × n_hyp_spk)
        ref_spk_ids: 参考中的说话人 ID 列表（去 -1）
        hyp_spk_ids: 预测中的说话人 ID 列表（去 -1）
    """
    ref_set = sorted(set(l for l in ref_labels if l >= 0))
    hyp_set = sorted(set(l for l in hyp_labels if l >= 0))

    if not ref_set or not hyp_set:
        return None, ref_set, hyp_set

    confusion = np.zeros((len(ref_set), len(hyp_set)), dtype=int)

    ref_to_idx = {s: i for i, s in enumerate(ref_set)}
    hyp_to_idx = {s: i for i, s in enumerate(hyp_set)}

    for r, h in zip(ref_labels, hyp_labels):
        if r >= 0 and h >= 0:
            confusion[ref_to_idx[r], hyp_to_idx[h]] += 1

    return confusion, ref_set, hyp_set


def optimal_mapping(ref_labels, hyp_labels):
    """
    使用匈牙利算法找到参考→预测的最佳说话人映射。

    返回:
        mapping: dict {ref_spk_id: hyp_spk_id}
        accuracy: 有声段的帧级准确率
    """
    confusion, ref_set, hyp_set = compute_confusion_matrix(ref_labels, hyp_labels)

    if confusion is None:
        return {}, 0.0

    n_ref = len(ref_set)
    n_hyp = len(hyp_set)

    # 匈牙利算法需要方阵；填充较小的一侧
    max_n = max(n_ref, n_hyp)
    cost = np.zeros((max_n, max_n))
    cost[:n_ref, :n_hyp] = -confusion  # 匈牙利求最小成本，我们要求最大匹配

    row_ind, col_ind = linear_sum_assignment(cost[:n_ref, :n_hyp] if n_ref <= n_hyp else np.pad(cost[:n_ref, :n_hyp], ((0, 0), (0, n_ref - n_hyp))))

    # 行=ref, 列=hyp
    mapping = {}
    for r_idx, h_idx in zip(row_ind, col_ind):
        if r_idx < n_ref and h_idx < n_hyp:
            mapping[ref_set[r_idx]] = hyp_set[h_idx]

    # 计算准确率
    correct = 0
    total = 0
    for r, h in zip(ref_labels, hyp_labels):
        if r >= 0 and h >= 0:
            total += 1
            if mapping.get(r) == h:
                correct += 1

    accuracy = correct / total if total > 0 else 0.0
    return mapping, accuracy


def evaluate_one(ref_labels, hyp_labels, ref_n_speakers):
    """
    评估单段音频的说话人分离效果。

    参数:
        ref_labels:     来自 RTTM 的帧级标签 (list of int, -1=静音)
        hyp_labels:     来自模型的帧级标签 (list of int, -1=静音)
        ref_n_speakers: RTTM 中的说话人数

    返回:
        dict 包含各项指标
    """
    # 对齐长度
    n = min(len(ref_labels), len(hyp_labels))
    ref = ref_labels[:n]
    hyp = hyp_labels[:n]

    # 最佳映射
    mapping, frame_accuracy = optimal_mapping(ref, hyp)

    # 应用映射后计算详细指标
    mapped_hyp = [mapping.get(r, h) if r >= 0 else h for r, h in zip(ref, hyp)]

    # 统计
    total_voice = sum(1 for r in ref if r >= 0)
    correct_voice = sum(1 for r, h in zip(ref, mapped_hyp) if r >= 0 and r == h)
    total_pred = sum(1 for h in hyp if h >= 0)

    # 漏检 (Miss): ref 有声，hyp 静音
    miss = sum(1 for r, h in zip(ref, hyp) if r >= 0 and h < 0)
    # 虚警 (False Alarm): ref 静音，hyp 有声
    fa = sum(1 for r, h in zip(ref, hyp) if r < 0 and h >= 0)
    # 说话人混淆: ref 和 hyp 都有声但人不对
    confusion_count = sum(1 for r, h in zip(ref, hyp) if r >= 0 and h >= 0 and mapping.get(r) != h)

    # DER = (Miss + FA + Confusion) / Total Ref Voice
    der = (miss + fa + confusion_count) / total_voice if total_voice > 0 else 1.0

    # 说话人数准确率
    hyp_n_speakers = len(set(h for h in hyp if h >= 0))
    n_spk_correct = (hyp_n_speakers == ref_n_speakers)

    return {
        'file_id': '',
        'total_frames': n,
        'voice_frames': total_voice,
        'ref_n_speakers': ref_n_speakers,
        'hyp_n_speakers': hyp_n_speakers,
        'n_spk_correct': n_spk_correct,
        'frame_accuracy': round(frame_accuracy * 100, 2),
        'DER': round(der * 100, 2),
        'miss': miss,
        'false_alarm': fa,
        'speaker_confusion': confusion_count,
        'mapping': mapping,
    }
