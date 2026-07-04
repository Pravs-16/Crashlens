"""Flask dashboard for CrashLens.

Renders the anomaly score over the test period, flagged segments with their
metric contribution scores, and the root-cause explanations. Run the
pipeline first (train -> detect -> explain), then:

    python app.py
"""

import base64
import io
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from flask import Flask, render_template_string

from crashlens import config

app = Flask(__name__)

PAGE = """
<!doctype html>
<html>
<head>
<title>CrashLens — {{ machine }}</title>
<style>
  body { font-family: system-ui, sans-serif; margin: 2rem auto; max-width: 1100px; color: #222; }
  h1 { margin-bottom: 0; }
  .sub { color: #666; margin-top: 0.2rem; }
  .metrics-row { display: flex; gap: 1rem; margin: 1.2rem 0; }
  .card { border: 1px solid #ddd; border-radius: 8px; padding: 0.8rem 1.2rem; }
  .card .value { font-size: 1.5rem; font-weight: 600; }
  .card .label { color: #666; font-size: 0.85rem; }
  img { max-width: 100%; border: 1px solid #eee; border-radius: 8px; }
  .segment { border: 1px solid #ddd; border-radius: 8px; padding: 1rem 1.2rem; margin: 1rem 0; }
  .segment.true { border-left: 5px solid #2a9d3a; }
  .segment.false { border-left: 5px solid #cc8400; }
  .bar-track { background: #eee; border-radius: 4px; height: 14px; width: 260px; display: inline-block; vertical-align: middle; }
  .bar { background: #c0392b; height: 14px; border-radius: 4px; }
  .metric-name { display: inline-block; width: 140px; font-family: monospace; }
  .explanation { background: #f7f7f7; border-radius: 6px; padding: 0.7rem 1rem; margin-top: 0.7rem; white-space: pre-wrap; }
  .tag { font-size: 0.75rem; padding: 2px 8px; border-radius: 10px; color: white; }
  .tag.true { background: #2a9d3a; } .tag.false { background: #cc8400; }
</style>
</head>
<body>
<h1>CrashLens</h1>
<p class="sub">Autoencoder anomaly detection on {{ machine }} (SMD) with metric-level contribution scores</p>

<div class="metrics-row">
  <div class="card"><div class="value">{{ f1 }}</div><div class="label">F1 (point-adjusted)</div></div>
  <div class="card"><div class="value">{{ precision }}</div><div class="label">Precision</div></div>
  <div class="card"><div class="value">{{ recall }}</div><div class="label">Recall</div></div>
  <div class="card"><div class="value">{{ hr100 }}</div><div class="label">HitRate@100% (contributions)</div></div>
  <div class="card"><div class="value">{{ n_segments }}</div><div class="label">Flagged segments</div></div>
</div>

<img src="data:image/png;base64,{{ plot }}" alt="anomaly score over time">

<h2>Flagged anomaly segments</h2>
{% for s in segments %}
<div class="segment {{ 'true' if s.overlaps_true_anomaly else 'false' }}">
  <b>Segment {{ s.id }}</b> &mdash; minutes {{ s.start }}&ndash;{{ s.end }}
  ({{ s.duration_min }} min)
  <span class="tag {{ 'true' if s.overlaps_true_anomaly else 'false' }}">
    {{ 'labeled anomaly' if s.overlaps_true_anomaly else 'unlabeled' }}
  </span>
  <div style="margin-top: 0.6rem;">
    {% for m in s.top_metrics %}
    <div>
      <span class="metric-name">{{ m.metric }}</span>
      <span class="bar-track"><span class="bar" style="width: {{ m.width }}%; display:block;"></span></span>
      <span style="font-family: monospace;">{{ m.score }}</span>
    </div>
    {% endfor %}
  </div>
  {% if s.explanation %}<div class="explanation">{{ s.explanation }}</div>{% endif %}
</div>
{% endfor %}
</body>
</html>
"""


def score_plot(npz):
    score, threshold, labels = npz["score"], float(npz["threshold"]), npz["labels"]
    fig, ax = plt.subplots(figsize=(12, 3.5))
    ax.plot(score, lw=0.6, color="#33507a", label="anomaly score")
    ax.axhline(threshold, color="#c0392b", lw=1, ls="--", label="threshold")
    ymax = max(score.max(), threshold) * 1.05
    ax.fill_between(np.arange(len(labels)), 0, ymax, where=labels > 0,
                    color="#2a9d3a", alpha=0.18, label="labeled anomaly")
    ax.set_ylim(0, ymax)
    ax.set_xlim(0, len(score))
    ax.set_xlabel("minute")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110)
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


@app.route("/")
def index():
    results = json.loads((config.ARTIFACT_DIR / "results.json").read_text())
    npz = np.load(config.ARTIFACT_DIR / "detection.npz")

    segments = results["segments"]
    for s in segments:
        top = max((m["score"] for m in s["top_metrics"]), default=1.0)
        for m in s["top_metrics"]:
            m["width"] = max(2, round(100 * max(m["score"], 0) / max(top, 1e-9)))

    det = results["detection"]
    return render_template_string(
        PAGE,
        machine=results["machine"],
        plot=score_plot(npz),
        f1=f"{det['f1']:.3f}",
        precision=f"{det['precision']:.3f}",
        recall=f"{det['recall']:.3f}",
        hr100=f"{results['interpretation']['hitrate@100%']:.3f}",
        n_segments=results["n_predicted_segments"],
        segments=segments,
    )


if __name__ == "__main__":
    app.run(debug=True, port=5000)
