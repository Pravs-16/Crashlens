"""Turn flagged anomalies into plain-English root-cause hypotheses.

Uses Claude when an Anthropic API key is available; otherwise falls back to
a deterministic rule-based explanation so the pipeline works offline. The
LLM only ever sees a small numeric summary of the anomaly (top contributing
metrics + trends), never raw telemetry — that keeps the prompt small and
the explanation grounded in the detector's own evidence.
"""

import argparse
import json

import numpy as np

from . import config
from .data import load_machine

MODEL = "claude-opus-4-8"

SYSTEM_PROMPT = """You are an SRE assistant analysing server telemetry anomalies.
You receive a summary of an anomaly window detected by an autoencoder-based
pipeline: which metrics contributed most (peak reconstruction error over the
window) and how those metrics moved relative to their normal baseline.

Respond with:
1. A one-sentence plain-English headline of the most likely root cause.
2. A short explanation (2-4 sentences) tying together the top metrics.
3. Two or three concrete checks an engineer should run next.

Ground every claim in the metrics provided. If the evidence is ambiguous,
say so rather than inventing a cause."""

# coarse metric categories used by the rule-based fallback
CATEGORIES = {
    "compute saturation (CPU/load)": ["cpu_r", "load_1", "load_5", "load_15"],
    "memory pressure / swapping": ["mem_shmem", "mem_u", "mem_u_e",
                                   "total_mem", "si", "so"],
    "disk I/O contention": ["disk_q", "disk_r", "disk_rb", "disk_svc",
                            "disk_u", "disk_w", "disk_wa", "disk_wb"],
    "network / connection issues": [
        "eth1_fi", "eth1_fo", "eth1_pi", "eth1_po", "tcp_tw", "tcp_use",
        "active_opens", "curr_estab", "in_errs", "in_segs",
        "listen_overflows", "out_rsts", "out_segs", "passive_opens",
        "retransegs", "tcp_timeouts", "udp_in_dg", "udp_out_dg",
        "udp_rcv_buf_errs", "udp_snd_buf_errs",
    ],
}


def build_report(segment, test_raw, train_mean, train_std):
    """Compact numeric summary of one anomaly for the LLM prompt."""
    start, end = segment["start"], segment["end"]
    lines = [
        f"Anomaly window: minutes {start}-{end} "
        f"({segment['duration_min']} min) on {config.MACHINE}",
        "Top contributing metrics (contribution = peak reconstruction "
        "error over the window; level shift vs normal baseline):",
    ]
    for entry in segment["top_metrics"]:
        m = config.METRIC_NAMES.index(entry["metric"])
        window_mean = test_raw[start:end + 1, m].mean()
        shift = (window_mean - train_mean[m]) / train_std[m]
        lines.append(
            f"- {entry['metric']}: contribution {entry['score']:.3f}, "
            f"window level {shift:+.1f} standard deviations vs baseline"
        )
    return "\n".join(lines)


def explain_rule_based(segment):
    """Deterministic fallback: map top metrics to coarse failure categories."""
    votes = {}
    for entry in segment["top_metrics"]:
        for category, metrics in CATEGORIES.items():
            if entry["metric"] in metrics:
                votes[category] = votes.get(category, 0.0) + max(entry["score"], 0.0)
    if not votes:
        return "No dominant metric category; inspect the window manually."
    top = max(votes, key=votes.get)
    metrics = ", ".join(e["metric"] for e in segment["top_metrics"][:3])
    return (f"Likely {top}: the metrics deviating most from normal behaviour "
            f"in this window were {metrics}. (rule-based explanation — set "
            f"ANTHROPIC_API_KEY for an LLM analysis)")


def explain_with_llm(client, report):
    response = client.messages.create(
        model=MODEL,
        max_tokens=1024,
        thinking={"type": "adaptive"},
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": report}],
    )
    return "\n".join(b.text for b in response.content if b.type == "text")


def get_client():
    try:
        import anthropic
        client = anthropic.Anthropic()
        # cheap auth check so we fail over to the rule-based path early
        client.models.retrieve(MODEL)
        return client
    except Exception:
        return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--segment", type=int, default=None,
                        help="explain only this segment id (default: all)")
    args = parser.parse_args()

    results = json.loads((config.ARTIFACT_DIR / "results.json").read_text())
    train_raw, test_raw, _ = load_machine()
    train_mean = train_raw.mean(axis=0)
    train_std = train_raw.std(axis=0) + 1e-8

    client = get_client()
    print("explainer:", "Claude" if client else "rule-based fallback")

    for segment in results["segments"]:
        if args.segment is not None and segment["id"] != args.segment:
            continue
        report = build_report(segment, test_raw, train_mean, train_std)
        if client:
            explanation = explain_with_llm(client, report)
        else:
            explanation = explain_rule_based(segment)
        segment["explanation"] = explanation
        print(f"\n--- segment {segment['id']} "
              f"({segment['start']}-{segment['end']}) ---")
        print(explanation)

    (config.ARTIFACT_DIR / "results.json").write_text(
        json.dumps(results, indent=2))
    print("\nexplanations saved to artifacts/results.json")


if __name__ == "__main__":
    main()
