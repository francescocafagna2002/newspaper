"""Run the battery pipeline end to end.

    python -m models.battery.pipeline              # everything, incl. the 77 GB scan
    python -m models.battery.pipeline --skip-scan  # reuse features_monthly.pkl

Steps 2 and 5 (labels, universe) are gates: if one fails the run stops with a
non-zero exit rather than adjusting a threshold to make it pass.
"""
from __future__ import annotations

import argparse
import time

from . import (build_features, build_features_agg, build_labels,
               define_universe, evaluate, score, train)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-scan", action="store_true",
                    help="reuse an existing features_monthly.pkl / features_daily.pkl")
    ap.add_argument("--months", nargs="*", help="limit the scan to these months")
    a = ap.parse_args()

    steps: list[tuple[str, callable]] = [
        ("1 labels", build_labels.main),
    ]
    if not a.skip_scan:
        steps.append(("3 features (streaming pass)", lambda: build_features.run(a.months)))
    steps += [
        ("2 universe", define_universe.main),
        ("4 aggregate", build_features_agg.main),
        ("5 train", train.train),
        ("6 evaluate", evaluate.main),
        ("7 score", score.main),
    ]

    for name, fn in steps:
        print(f"\n{'=' * 70}\n== {name}\n{'=' * 70}")
        t0 = time.time()
        fn()
        print(f"-- {name} done in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
