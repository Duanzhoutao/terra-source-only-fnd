from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, f1_score, recall_score, roc_auc_score


def clipped_logit(prob: np.ndarray) -> np.ndarray:
    p = np.asarray(prob, dtype=np.float64).clip(1e-6, 1.0 - 1e-6)
    return np.log(p / (1.0 - p))


def sigmoid(value: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.asarray(value, dtype=np.float64)))


def logit_blend(primary: np.ndarray, auxiliary: np.ndarray, aux_weight: float) -> np.ndarray:
    weight = float(aux_weight)
    out = (1.0 - weight) * clipped_logit(primary) + weight * clipped_logit(auxiliary)
    return sigmoid(out).clip(1e-6, 1.0 - 1e-6)


def ece_binary(y_true: np.ndarray, score: np.ndarray, bins: int = 10) -> float:
    labels = np.asarray(y_true, dtype=np.float64)
    probs = np.asarray(score, dtype=np.float64).clip(0.0, 1.0)
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = len(probs)
    if total == 0:
        return float("nan")
    ece = 0.0
    for idx in range(bins):
        if idx == bins - 1:
            mask = (probs >= edges[idx]) & (probs <= edges[idx + 1])
        else:
            mask = (probs >= edges[idx]) & (probs < edges[idx + 1])
        count = int(mask.sum())
        if count:
            ece += (count / total) * abs(float(probs[mask].mean()) - float(labels[mask].mean()))
    return float(ece)


def choose_threshold(y_true: np.ndarray, score: np.ndarray, rule: str = "balanced_acc") -> tuple[float, float]:
    labels = np.asarray(y_true, dtype=np.int64)
    scores = np.asarray(score, dtype=np.float64)
    if labels.size == 0:
        return 0.5, 0.0

    order = np.argsort(scores, kind="mergesort")[::-1]
    sorted_scores = scores[order]
    sorted_labels = labels[order]
    total_pos = float(sorted_labels.sum())
    total_neg = float(len(sorted_labels) - sorted_labels.sum())
    tp = np.cumsum(sorted_labels == 1).astype(np.float64)
    fp = np.cumsum(sorted_labels == 0).astype(np.float64)
    change = np.r_[np.diff(sorted_scores) != 0, True]
    idx = np.flatnonzero(change)
    tp = tp[idx]
    fp = fp[idx]
    thresholds = sorted_scores[idx]
    fn = total_pos - tp
    tn = total_neg - fp
    fake_recall = tp / np.maximum(total_pos, 1.0)
    real_recall = tn / np.maximum(total_neg, 1.0)
    fake_precision = tp / np.maximum(tp + fp, 1.0)
    real_precision = tn / np.maximum(tn + fn, 1.0)
    fake_f1 = 2.0 * fake_precision * fake_recall / np.maximum(fake_precision + fake_recall, 1e-12)
    real_f1 = 2.0 * real_precision * real_recall / np.maximum(real_precision + real_recall, 1e-12)
    macro_f1 = 0.5 * (fake_f1 + real_f1)
    balanced_acc = 0.5 * (fake_recall + real_recall)

    if rule == "balanced_acc":
        primary = balanced_acc
    elif rule == "macro_f1":
        primary = macro_f1
    else:
        raise ValueError(f"Unsupported threshold rule: {rule}")
    best = int(np.lexsort((macro_f1, primary))[-1])
    return float(thresholds[best]), float(macro_f1[best])


def binary_metrics(y_true: np.ndarray, score: np.ndarray, threshold: float) -> dict[str, float]:
    labels = np.asarray(y_true, dtype=int)
    probs = np.asarray(score, dtype=np.float64).clip(1e-6, 1.0 - 1e-6)
    pred = (probs >= float(threshold)).astype(int)
    try:
        auc = float(roc_auc_score(labels, probs))
    except ValueError:
        auc = float("nan")
    return {
        "macro_f1": float(f1_score(labels, pred, average="macro", zero_division=0)),
        "auc": auc,
        "fake_recall": float(recall_score(labels, pred, pos_label=1, zero_division=0)),
        "fake_auprc": float(average_precision_score(labels, probs)),
        "ece_10bin_fake_prob": ece_binary(labels, probs, bins=10),
    }

