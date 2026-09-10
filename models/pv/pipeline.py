"""Run the whole PV-detection pipeline end to end.

    python -m newspaper.models.pv.pipeline              # full run (~37 min, one 77 GB pass)
    python -m newspaper.models.pv.pipeline --skip-scan  # reuse features_monthly.pkl

Stages: prep -> features -> modeling. See ../../docs/pv_mvp_plan.md and REPORT.md.
"""
from __future__ import annotations

import argparse
import time

from .prep import labels, silver_labels, weather, base_rate
from .features import extract, aggregate
from .modeling import train, evaluate, score


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-scan", action="store_true",
                    help="reuse an existing features_monthly.pkl / feedin_meter_stats.csv")
    args = ap.parse_args()

    steps = [
        ("prep.labels", labels.main),
        ("prep.weather", weather.main),
        ("prep.base_rate", base_rate.main),
    ]
    if not args.skip_scan:
        steps.append(("features.extract (77 GB streaming pass)", lambda: extract.run(None)))
    steps += [
        ("prep.silver_labels", silver_labels.main),
        ("features.aggregate", aggregate.main),
        ("modeling.train", train.train),
        ("modeling.evaluate", evaluate.main),
        ("modeling.score", score.main),
    ]
    for name, fn in steps:
        print(f"\n{'=' * 70}\n=== {name}\n{'=' * 70}")
        t = time.time()
        fn()
        print(f"--- {name}: {time.time() - t:.1f}s")


if __name__ == "__main__":
    main()
