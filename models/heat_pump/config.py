"""Heat-pump-classifier paths, reusing the PV pipeline's I/O contract."""
from __future__ import annotations

from ..pv import config as BASE

NEWSPAPER = BASE.NEWSPAPER
DATA_ADDITION = BASE.DATA_ADDITION
CSV_SEP = BASE.CSV_SEP
SEED = 20260911

TRAIN_SOURCE = DATA_ADDITION / "train_property_samples_auxw_netload.csv"
WPB_HOUSEHOLD_PREDICTIONS = (
    NEWSPAPER / "models" / "heat_pump_boiler" / "artifacts" / "wpb_household_predictions.csv"
)

ARTIFACTS = NEWSPAPER / "models" / "heat_pump" / "artifacts"
FIGURES = ARTIFACTS / "figures"
FEATURES = ARTIFACTS / "heat_pump_features.csv"

for _path in (ARTIFACTS, FIGURES):
    _path.mkdir(parents=True, exist_ok=True)
