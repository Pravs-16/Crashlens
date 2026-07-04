"""Download the Server Machine Dataset (SMD) subset used by this project.

SMD comes from the OmniAnomaly paper (Su et al., KDD 2019) and contains
5 weeks of telemetry (38 metrics, 1-minute resolution) from production
servers at a large internet company, with labeled anomalies and
per-anomaly interpretation labels (which metrics caused each anomaly).

Source: https://github.com/NetManAIOps/OmniAnomaly (ServerMachineDataset/)
"""

import argparse
import urllib.request
from pathlib import Path

BASE_URL = (
    "https://raw.githubusercontent.com/NetManAIOps/OmniAnomaly/master/"
    "ServerMachineDataset"
)

PARTS = ["train", "test", "test_label", "interpretation_label"]


def download(machine: str, out_dir: Path) -> None:
    for part in PARTS:
        target = out_dir / part / f"{machine}.txt"
        if target.exists():
            print(f"already present: {target}")
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        url = f"{BASE_URL}/{part}/{machine}.txt"
        print(f"downloading {url}")
        urllib.request.urlretrieve(url, target)
        print(f"  -> {target} ({target.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--machine", default="machine-1-1",
                        help="SMD machine id, e.g. machine-1-1")
    parser.add_argument("--out", default="data/smd",
                        help="output directory")
    args = parser.parse_args()
    download(args.machine, Path(args.out))
