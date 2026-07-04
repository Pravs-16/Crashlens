# Design Decisions

A log of the non-obvious choices in this project and the reasoning behind
them. Every one of these is a question an interviewer could reasonably ask.

## Why SMD (Server Machine Dataset)?

I needed public telemetry that looks like enterprise device/server telemetry:
multivariate (CPU, memory, disk I/O, network), minute-resolution, with
labeled anomalies. SMD (from the OmniAnomaly paper, KDD 2019) fits, and it
has one property almost no other anomaly benchmark has: **interpretation
labels** — for every anomaly, the ground-truth set of metrics that caused it.
That means my metric contribution scores can be evaluated quantitatively
(HitRate@k) instead of eyeballed.

Alternatives considered:
- **NAB (Numenta)** — real cloud telemetry, but mostly univariate series, so
  no per-metric contribution story.
- **Loghub (HDFS/BGL logs)** — log lines, not metrics; better suited to a
  log-parsing project than a telemetry one.

## Why an autoencoder instead of simpler methods?

Two candidate reasons going in — and running the baselines
(`python -m crashlens.baselines`, identical evaluation protocol) showed
only one of them actually holds on this machine:

1. **Joint structure / detection quality.** The theory: a per-metric
   z-score treats metrics independently, while the autoencoder learns their
   joint distribution, so correlation violations raise the error. In
   practice, on machine-1-1 under the point-adjust protocol, the z-score
   baseline scores F1 0.996 vs the autoencoder's 0.981 — the protocol is so
   forgiving (one detected point credits the whole segment) that every
   method saturates. Detection alone does **not** justify the neural model
   here, and I say so rather than hiding the baseline row.
2. **Per-metric errors for free.** This is the real justification.
   Reconstruction error decomposes naturally per feature, which is what
   makes the contribution scores (HitRate@100% ≈ 0.74 against ground truth)
   and the downstream explanation agent possible. The z-score baseline's
   per-metric values exist but ignore correlations; Isolation Forest gives
   one opaque score per point and needs extra machinery (e.g. SHAP) for any
   attribution at all.

## Why a pointwise dense autoencoder, not an LSTM/sequence model?

Started with the simplest model that can capture cross-metric structure.
A pointwise AE (one timestep in, one timestep out) already competes with
much heavier sequence models on SMD in the literature, is fast to train on
CPU, and every design element is explainable. Sequence models (LSTM-AE,
OmniAnomaly's stochastic RNN) capture temporal patterns too and are the
natural next step — but adding that complexity before validating the simple
version would have been premature.

## Why train on normal data only?

Reconstruction-based detection rests on this: the model only learns to
reconstruct "normal", so anything abnormal reconstructs poorly. SMD's train
split is anomaly-free by construction. In a production setting you'd need to
curate a mostly-clean training window instead.

## Why a chronological (not random) train/validation split?

Adjacent timesteps in telemetry are nearly identical. A random split leaks
near-duplicates of training points into validation, making the val loss
optimistically biased and early stopping meaningless. Holding out the last
10% chronologically avoids that.

## Why min-max scaling fit on train only, with clipping?

Scaling must be fit on training data only — using test statistics leaks
information. Test values are allowed to leave [0, 1] (that deviation *is*
the signal), but a few SMD metrics are near-constant in training; without a
floor on the denominator and a clip on the output, those metrics explode at
test time and drown out everything else.

## How the contribution scores were chosen (median vs. peak — an actual experiment)

The first design was **median** per-metric error over the anomaly window
(robust: one spiky timestep can't dominate) **z-scored** against each
metric's error distribution on normal training data (comparable across
metrics whose error scales differ by orders of magnitude). Plausible on
paper — but SMD's interpretation labels made it testable, and it tested
badly. Grid over aggregation × normalization on machine-1-1:

| Aggregation | Normalization | HitRate@100% | HitRate@150% |
|---|---|---|---|
| median | z-score | 0.31 | 0.36 |
| median | raw | 0.36 | 0.46 |
| 90th percentile | raw | 0.57 | 0.67 |
| errors at peak timestep | raw | 0.62 | 0.72 |
| **max (peak error)** | **raw** | **0.74** | **0.81** |

Two lessons:

1. **Median dilutes attribution.** Labeled incidents on this machine run up
   to 722 minutes, and the causal metrics only deviate during part of that
   window — over the full window their *median* error looks ordinary, so
   the ranking degrades. Peak error asks "which metric deviated hardest at
   any point in the incident", which matches how the labels were assigned.
   (Rolling-median *smoothing* before the max was also tried — it helped
   nothing and destroyed the short 3–4 minute incidents at larger widths.)
2. **Z-scoring the errors hurt ranking.** Metrics that are near-constant in
   training have vanishing error std, so the z-score explodes on them
   spuriously; flooring the std helps but raw peak error still ranked best.
   Cross-metric comparability is already largely handled by min-max scaling
   the inputs.

Spike robustness didn't disappear: the *detection* score (mean error across
metrics) is still rolling-median smoothed before thresholding, so isolated
one-minute glitches don't trip the detector. The median's job moved to the
detection path; attribution uses peak error.

## Why point-adjusted F1 and best-F1 threshold search?

Standard evaluation protocol on SMD (introduced by Xu et al. 2018, used by
OmniAnomaly, USAD, etc.): if any point inside a true anomaly segment is
detected, the whole segment counts — operators care about catching the
incident, not flagging every minute of it. Reporting F1 at the best
threshold keeps numbers comparable to the literature. In production you
don't get to peek at labels: you'd set the threshold from a percentile of
training reconstruction errors (e.g. 99.5th) and tune it against the
operational false-alarm budget.

## Why MongoDB and not a vector database?

The persistence need is: heterogeneous documents (anomaly reports with
nested metric rankings and free-text explanations) queried by machine id and
time range. That's a document-store workload. There is no similarity search
anywhere in this pipeline, so a vector DB would solve a problem that doesn't
exist. If I later add "retrieve similar past incidents" for the explanation
agent, embeddings + a vector index become justified — that's the criterion.

## Why does the LLM only see a numeric summary, not raw telemetry?

Three reasons: (1) grounding — the explanation should be derived from the
detector's own evidence (top contributing metrics + baseline deviations),
not from the LLM re-analysing raw numbers it's bad at; (2) cost/latency — a
window of raw 38-dim telemetry is thousands of tokens for no benefit;
(3) reproducibility — the summary makes each explanation auditable.

The rule-based fallback exists so the pipeline is runnable and demoable
without an API key, and it doubles as a sanity check on the LLM output
(both should point at the same metric category).

## Known limitations (asked about these too — be honest)

- Threshold selection uses test labels (best-F1 protocol). Fine for
  benchmarking, not deployable as-is.
- Pointwise model ignores temporal ordering within the window; slow drifts
  that stay inside the normal per-timestep envelope can slip through.
- Trained/evaluated on one machine (machine-1-1); no cross-machine
  generalization claim.
- SMD anomalies are server incidents, not literal app crashes — the method
  transfers, the labels differ.
