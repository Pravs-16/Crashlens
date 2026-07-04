"""Evaluation: point-adjusted detection metrics and interpretation HitRate."""

import numpy as np


def label_segments(labels):
    """Contiguous runs of 1s in a binary array -> list of (start, end_incl)."""
    segments, start = [], None
    for i, v in enumerate(labels):
        if v and start is None:
            start = i
        elif not v and start is not None:
            segments.append((start, i - 1))
            start = None
    if start is not None:
        segments.append((start, len(labels) - 1))
    return segments


def point_adjust(pred, labels):
    """Point-adjust protocol (Xu et al., WWW 2018): if any point inside a
    true anomaly segment is predicted, the whole segment counts as found.
    Rationale: operators care about catching the incident, not every minute
    of it."""
    adjusted = pred.copy()
    for start, end in label_segments(labels):
        if adjusted[start:end + 1].any():
            adjusted[start:end + 1] = 1
    return adjusted


def prf(pred, labels):
    tp = int(((pred == 1) & (labels == 1)).sum())
    fp = int(((pred == 1) & (labels == 0)).sum())
    fn = int(((pred == 0) & (labels == 1)).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def best_f1(scores, labels, n_thresholds=400):
    """Search score quantiles for the threshold with the best point-adjusted
    F1. Reporting best F1 is the standard protocol on SMD (OmniAnomaly,
    USAD, ...) and keeps results comparable to the literature."""
    candidates = np.quantile(scores, np.linspace(0.5, 0.9999, n_thresholds))
    best = {"f1": -1.0}
    for th in np.unique(candidates):
        pred = point_adjust((scores >= th).astype(int), labels)
        p, r, f1 = prf(pred, labels)
        if f1 > best["f1"]:
            best = {"threshold": float(th), "precision": p, "recall": r, "f1": f1}
    return best


def hitrate(contrib_ranking, gt_dims, percent):
    """HitRate@P%: fraction of ground-truth contributing metrics found in
    the top ceil(P% * |GT|) of the predicted contribution ranking."""
    k = int(np.ceil(percent / 100 * len(gt_dims)))
    return len(set(contrib_ranking[:k]) & set(gt_dims)) / len(gt_dims)
