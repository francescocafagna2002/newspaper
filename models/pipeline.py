"""Run the whole PV-detection pipeline end to end.

    python -m newspaper.models.pipeline           # full run (~40 min, one 77 GB pass)
    python -m newspaper.models.pipeline --skip-scan   # reuse features_monthly.pkl

Steps 1-10 of ../docs/pv_mvp_plan.md.
"""
from __future__ import annotations

import argparse
import time

from . import (build_labels, build_weather, build_baserate, build_features,
               extract_feedin_labels, build_features_agg, train, evaluate, score)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-scan", action="store_true",
                    help="reuse an existing features_monthly.pkl / feedin_meter_stats.csv")
    args = ap.parse_args()

    steps = [
        ("build_labels", build_labels.main),
        ("build_weather", build_weather.main),
        ("build_baserate", build_baserate.main),
    ]
    if not args.skip_scan:
        steps.append(("build_features (77 GB streaming pass)", lambda: build_features.run(None)))
    steps += [
        ("extract_feedin_labels", extract_feedin_labels.main),
        ("build_features_agg", build_features_agg.main),
        ("train", train.train),
        ("evaluate", evaluate.main),
        ("score", score.main),
    ]
    for name, fn in steps:
        print(f"\n{'=' * 70}\n=== {name}\n{'=' * 70}")
        t = time.time()
        fn()
        print(f"--- {name}: {time.time() - t:.1f}s")


if __name__ == "__main__":
    main()
