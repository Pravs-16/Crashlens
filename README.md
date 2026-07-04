# CrashLens — Explainable Telemetry Anomaly Detection

An unsupervised anomaly detection pipeline for server/application telemetry
that not only flags anomalous windows, but **scores which metrics contributed**
to each anomaly and uses an LLM agent to produce a plain-English root-cause
hypothesis with suggested next checks.

Built on the [Server Machine Dataset (SMD)](https://github.com/NetManAIOps/OmniAnomaly)
— 38 telemetry metrics (CPU, load, memory, disk I/O, swap, network) at
1-minute resolution from production servers, with labeled anomalies **and**
ground-truth interpretation labels (which metrics caused each anomaly), so
the contribution scores are validated quantitatively, not eyeballed.

## Architecture

```
SMD telemetry (train = normal only, test = labeled)
        │
        ▼
  Preprocessing ── min-max scaling fit on train, clipped
        │
        ▼
  Dense autoencoder (PyTorch, 38 → 24 → 8 → 24 → 38)
  trained on normal data; reconstruction error = anomaly signal
        │
        ├─► overall score = mean per-metric error, rolling-median smoothed
        │       └─► threshold ─► flagged anomaly segments
        │
        └─► per-metric contribution scores
                = peak per-metric error over the window
                  (aggregation choice validated against SMD's
                   ground-truth interpretation labels — see DECISIONS.md)
        │
        ▼
  LLM explanation agent (Claude) ── plain-English root-cause hypothesis
  (rule-based fallback when no API key is configured)
        │
        ▼
  Flask dashboard + optional MongoDB persistence of anomaly reports
```

## Results (machine-1-1)

| Method | Precision | Recall | F1 (point-adjusted) |
|---|---|---|---|
| Per-metric z-score baseline | 0.995 | 0.997 | 0.996 |
| Isolation Forest baseline | 0.933 | 0.997 | 0.964 |
| Autoencoder (this repo) | 0.963 | 0.999 | 0.981 |

Contribution-score quality vs. SMD ground-truth interpretation labels
(fraction of the true causal metrics recovered in the top of the ranking):

| HitRate@100% | HitRate@150% |
|---|---|
| 0.742 | 0.813 |

Honest reading: on this machine the point-adjust protocol saturates — even
the simple z-score baseline reaches F1 ≈ 0.99, because catching one point
of a segment counts as catching the whole segment. Detection quality alone
doesn't justify the autoencoder here. What the baselines *can't* do is
attribution: the autoencoder's per-metric reconstruction errors recover
~74% of the ground-truth causal metrics per anomaly, which is what feeds
the explanation layer. That trade-off (and the aggregation experiment
behind the HitRate numbers) is documented in
[DECISIONS.md](DECISIONS.md).

*(Numbers from `python -m crashlens.baselines` and `python -m
crashlens.detect`; standard point-adjust protocol and best-F1 threshold
search, as used across the SMD literature.)*

## Quickstart

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows (source .venv/bin/activate on Linux)
pip install -r requirements.txt

python scripts/download_data.py    # fetch SMD machine-1-1 (~20 MB)
python -m crashlens.train          # train the autoencoder (CPU, ~1 min)
python -m crashlens.baselines      # z-score + Isolation Forest comparison
python -m crashlens.detect         # score test set, flag anomalies, contributions
python -m crashlens.explain        # LLM (or rule-based) root-cause explanations
python app.py                      # dashboard at http://localhost:5000
```

For LLM explanations, set an Anthropic API key first
(`set ANTHROPIC_API_KEY=...`); without one, a deterministic rule-based
explainer is used so the demo still works offline.

To persist anomaly reports to MongoDB (optional):

```bash
python -m crashlens.store          # upserts into crashlens.anomaly_reports
```

## Why these design choices?

Every non-obvious decision — why an autoencoder over Isolation Forest, why
median aggregation, why z-scored contributions, why MongoDB and not a vector
DB, why point-adjusted F1 — is documented with its reasoning in
[DECISIONS.md](DECISIONS.md), along with the known limitations.

## Project layout

```
crashlens/
  config.py      metric names, paths, hyperparameters
  data.py        SMD loading, interpretation-label parsing, scaling
  model.py       dense autoencoder
  train.py       training with chronological validation + early stopping
  detect.py      scoring, thresholding, segments, contribution scores
  evaluate.py    point-adjusted P/R/F1, best-F1 search, HitRate@k
  baselines.py   rolling z-score and Isolation Forest baselines
  explain.py     LLM / rule-based root-cause explanations
  store.py       optional MongoDB persistence
scripts/
  download_data.py
app.py           Flask dashboard
```

## Dataset

SMD is released with the OmniAnomaly paper: Su et al., *Robust Anomaly
Detection for Multivariate Time Series through Stochastic Recurrent Neural
Network*, KDD 2019. The download script pulls one machine's data
(train / test / test_label / interpretation_label) from the official repo.
