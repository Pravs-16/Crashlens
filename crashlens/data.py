"""Loading and preprocessing for SMD telemetry."""

import numpy as np

from . import config


def load_machine(machine=config.MACHINE, data_dir=config.DATA_DIR):
    train = np.loadtxt(data_dir / "train" / f"{machine}.txt", delimiter=",")
    test = np.loadtxt(data_dir / "test" / f"{machine}.txt", delimiter=",")
    labels = np.loadtxt(data_dir / "test_label" / f"{machine}.txt").astype(int)
    return train, test, labels


def load_interpretation(machine=config.MACHINE, data_dir=config.DATA_DIR):
    """Parse interpretation labels: lines like ``15849-16368:1,9,10``.

    Each line is an anomaly span (test-set indices) plus the 1-indexed ids
    of the metrics that caused it. Returns (start, end_inclusive, dims)
    tuples with 0-indexed metric ids.
    """
    path = data_dir / "interpretation_label" / f"{machine}.txt"
    segments = []
    for line in path.read_text().strip().splitlines():
        span, dims = line.split(":")
        start, end = (int(v) for v in span.split("-"))
        dim_ids = sorted(int(d) - 1 for d in dims.split(","))
        segments.append((start, end, dim_ids))
    return segments


class MinMaxScaler:
    """Min-max scaling fitted on training data only.

    Test values are allowed to leave [0, 1] — that deviation is signal —
    but are clipped to a bounded range so metrics that are near-constant
    during training cannot blow up to huge values and drown out the rest.
    """

    def __init__(self, clip=(-2.0, 4.0)):
        self.clip = clip

    def fit(self, x):
        self.min_ = x.min(axis=0)
        self.range_ = np.maximum(x.max(axis=0) - self.min_, 1e-3)
        return self

    def transform(self, x):
        return np.clip((x - self.min_) / self.range_, *self.clip)
