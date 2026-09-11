"""Add heat-pump-specific engineered features to the shared property-sample table.

`data_addition/Readme.md` already ran a generic HistGradientBoosting model on
`train_property_samples_auxw_netload.csv` for `has_Waermepumpe` and got
ROC-AUC 0.565 (near chance). `docs/data_problems.md` names two things the
generic aggregate hides: a household's own seasonality is confounded with its
size, and a *dedicated* heat-pump meter (present for only 29 households) is
diluted when meters are summed. This step adds:

  * `winter_share_over_night_share` - `winter_summer_ratio` normalised by
    `night_share` (a household-size proxy per `docs/data_problems.md`), so two
    households with the same absolute winter/summer swing but different sizes
    don't look identical.
  * `wpb_score` / `wpb_flag` - the `models.heat_pump_boiler` physics-based
    plateau detector's household-level output. It targets hot-water boilers,
    not space heating, but a household running one heat-pump appliance is more
    likely to run another, so its score is a legitimate (if imperfect) input
    feature rather than a target proxy.

A true temperature-coupling feature (heating-degree-days regressed against
each household's daily import, the PV-weather-coupling analogue) needs
per-household daily series. `train_property_sequences.csv` would supply that
but is not present in this checkout (`ev_classification_total/README.md`
confirms it was never in this checkout either) - `load_property_daily.csv` is
also absent. That feature is therefore not implemented; see the README's
limitations section rather than silently dropping it.

Output: artifacts/heat_pump_features.csv
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C


def main() -> pd.DataFrame:
    train = pd.read_csv(C.TRAIN_SOURCE, sep=C.CSV_SEP, dtype={"gp_nr": str, "plz": str})

    wpb = pd.read_csv(C.WPB_HOUSEHOLD_PREDICTIONS, dtype={"gp_nr": str, "plz": str})
    wpb = wpb[["gp_nr", "wpb_score", "wpb_flag"]].drop_duplicates("gp_nr")

    out = train.merge(wpb, on="gp_nr", how="left")
    n_matched = out["wpb_score"].notna().sum()
    out["wpb_score"] = out["wpb_score"].fillna(0.0)
    out["wpb_flag"] = out["wpb_flag"].fillna(False).astype(int)

    out["winter_share_over_night_share"] = out["winter_summer_ratio"] / out["night_share"].where(
        out["night_share"] > 0.01
    )
    out["winter_share_over_night_share"] = out["winter_share_over_night_share"].replace(
        [np.inf, -np.inf], np.nan
    )

    C.FEATURES.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(C.FEATURES, sep=C.CSV_SEP, index=False)

    print(f"wrote {C.FEATURES}  ({len(out)} rows, +3 cols)")
    print(f"  households matched to a WPB score : {n_matched}/{len(out)}")
    print(f"  households with wpb_flag=1        : {int(out['wpb_flag'].sum())}")
    hp = out[out["has_Waermepumpe"] == 1]
    rest = out[out["has_Waermepumpe"] == 0]
    # NOTE: has_Waermepumpe == 0 is unlabelled, not a verified negative
    # (docs/data_problems.md), so this contrast is a floor on separation.
    for c in ("wpb_score", "winter_share_over_night_share"):
        print(f"  {c:32s}: HP-labelled median {hp[c].median():.3f} "
              f"vs unlabelled median {rest[c].median():.3f}")
    return out


if __name__ == "__main__":
    main()
