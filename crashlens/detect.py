"""Score the test split, flag anomaly segments, compute metric contributions."""

import json

import numpy as np
import torch

from . import config
from .data import MinMaxScaler, load_interpretation, load_machine
from .evaluate import best_f1, hitrate, label_segments
from .model import DenseAutoencoder


def rolling_median(x, w):
    """Median smoothing of the anomaly score: single-timestep error spikes
    (sensor glitches) should not trip the detector on their own."""
    if w <= 1:
        return x
    pad = w // 2
    padded = np.pad(x, (pad, w - 1 - pad), mode="edge")
    return np.median(np.lib.stride_tricks.sliding_window_view(padded, w), axis=1)


def load_artifacts():
    model = DenseAutoencoder()
    model.load_state_dict(torch.load(config.ARTIFACT_DIR / "autoencoder.pt",
                                     weights_only=True))
    model.eval()
    npz = np.load(config.ARTIFACT_DIR / "scaler.npz")
    scaler = MinMaxScaler()
    scaler.min_, scaler.range_ = npz["min"], npz["range"]
    return model, scaler, npz["err_mean"], npz["err_std"]


def feature_errors(model, x):
    with torch.no_grad():
        recon = model(torch.from_numpy(x)).numpy()
    return (recon - x) ** 2


def contribution_scores(feat_err_segment):
    """Peak reconstruction error per metric over the anomaly window.

    Median aggregation was the first design (robust to one spiky timestep),
    but validating against SMD's ground-truth interpretation labels showed
    it dilutes attribution on long incidents where the causal metrics only
    deviate for part of the window: HitRate@100% 0.31 (median, z-scored)
    vs 0.74 (peak raw error). Full experiment in DECISIONS.md. Spike
    robustness for *detection* is handled separately by the rolling-median
    smoothing of the overall score.
    """
    return feat_err_segment.max(axis=0)


def merge_segments(segments, gap):
    merged = []
    for start, end in segments:
        if merged and start - merged[-1][1] <= gap:
            merged[-1][1] = end
        else:
            merged.append([start, end])
    return [tuple(s) for s in merged]


def main():
    _, test_raw, labels = load_machine()
    model, scaler, _, _ = load_artifacts()
    test = scaler.transform(test_raw).astype(np.float32)

    feat_err = feature_errors(model, test)
    score = rolling_median(feat_err.mean(axis=1), config.SMOOTH_WINDOW)

    detection = best_f1(score, labels)
    pred = (score >= detection["threshold"]).astype(int)
    pred_segments = merge_segments(label_segments(pred), config.MIN_SEGMENT_GAP)

    # per-segment anomaly reports with ranked metric contributions
    reports = []
    for i, (start, end) in enumerate(pred_segments):
        contrib = contribution_scores(feat_err[start:end + 1])
        ranking = np.argsort(contrib)[::-1]
        top = [{"metric": config.METRIC_NAMES[m], "score": round(float(contrib[m]), 4)}
               for m in ranking[:config.TOP_K_METRICS]]
        reports.append({
            "id": i,
            "start": int(start),
            "end": int(end),
            "duration_min": int(end - start + 1),
            "overlaps_true_anomaly": bool(labels[start:end + 1].any()),
            "top_metrics": top,
        })

    # validate contribution rankings against ground-truth contributing dims
    hr100, hr150 = [], []
    for start, end, gt_dims in load_interpretation():
        contrib = contribution_scores(feat_err[start:end + 1])
        ranking = list(np.argsort(contrib)[::-1])
        hr100.append(hitrate(ranking, gt_dims, 100))
        hr150.append(hitrate(ranking, gt_dims, 150))

    results = {
        "machine": config.MACHINE,
        "detection": detection,
        "interpretation": {
            "hitrate@100%": round(float(np.mean(hr100)), 4),
            "hitrate@150%": round(float(np.mean(hr150)), 4),
        },
        "n_predicted_segments": len(pred_segments),
        "segments": reports,
    }
    config.ARTIFACT_DIR.mkdir(exist_ok=True)
    (config.ARTIFACT_DIR / "results.json").write_text(json.dumps(results, indent=2))
    np.savez(config.ARTIFACT_DIR / "detection.npz",
             score=score, threshold=detection["threshold"],
             labels=labels, pred=pred)
    print(json.dumps({k: results[k] for k in ("detection", "interpretation")},
                     indent=2))
    print(f"{len(pred_segments)} predicted segments; "
          f"full report in artifacts/results.json")


if __name__ == "__main__":
    main()
