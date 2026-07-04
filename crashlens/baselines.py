"""Baselines the autoencoder has to beat: per-metric z-score and Isolation
Forest. If a simple method matched the autoencoder there would be no reason
to keep the neural model."""

import json

import numpy as np
from sklearn.ensemble import IsolationForest

from . import config
from .data import MinMaxScaler, load_machine
from .evaluate import best_f1


def zscore_baseline(train, test):
    """Max absolute z-score across metrics, relative to training statistics.

    Purely per-metric: it cannot see joint structure, which is exactly the
    weakness the autoencoder is supposed to address.
    """
    mean, std = train.mean(axis=0), train.std(axis=0) + 1e-8
    return np.abs((test - mean) / std).max(axis=1)


def isolation_forest_baseline(train, test):
    forest = IsolationForest(n_estimators=100, random_state=config.SEED)
    forest.fit(train)
    return -forest.score_samples(test)


def main():
    train_raw, test_raw, labels = load_machine()
    scaler = MinMaxScaler().fit(train_raw)
    train, test = scaler.transform(train_raw), scaler.transform(test_raw)

    results = {}
    for name, scores in [
        ("zscore", zscore_baseline(train, test)),
        ("isolation_forest", isolation_forest_baseline(train, test)),
    ]:
        results[name] = best_f1(scores, labels)
        print(name, json.dumps(results[name]))

    config.ARTIFACT_DIR.mkdir(exist_ok=True)
    (config.ARTIFACT_DIR / "baselines.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
