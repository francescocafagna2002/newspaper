"""Battery-specific paths and constants, reusing the PV pipeline I/O contract."""
from __future__ import annotations

from pathlib import Path

from ..pv import config as BASE

WORK = BASE.WORK
NEWSPAPER = BASE.NEWSPAPER
INPUT_DATA = BASE.INPUT_DATA
DATA_ADDITION = BASE.DATA_ADDITION
GIGI_ANNOTATED = BASE.GIGI_ANNOTATED
WEATHER_DAILY = BASE.WEATHER_DAILY
PV_BASERATE_PLZ = BASE.PV_BASERATE_PLZ
CSV_SEP = BASE.CSV_SEP
CSV_ENCODING = BASE.CSV_ENCODING
SEED = BASE.SEED
OBIS_IMPORT = BASE.OBIS_IMPORT
OBIS_EXPORT = BASE.OBIS_EXPORT
QUARTER_HOUR_COLS = BASE.QUARTER_HOUR_COLS
SUMMER_MONTHS = BASE.SUMMER_MONTHS
WINTER_MONTHS = BASE.WINTER_MONTHS
SHOULDER_MONTHS = (3, 4, 9, 10)
RUNLEN_MONTHS = tuple(range(3, 11))

ARTIFACTS = NEWSPAPER / "models" / "battery" / "artifacts"
MODEL_DIR = ARTIFACTS / "model"
FIGURES = ARTIFACTS / "figures"
LABELS = DATA_ADDITION / "battery_labels.csv"
FEATURES_MONTHLY = ARTIFACTS / "features_monthly.pkl"
DAILY_FEATURES = ARTIFACTS / "features_daily.pkl"
METER_EXPORT_DAYS = ARTIFACTS / "meter_export_days.csv"
UNIVERSE = ARTIFACTS / "universe.csv"
FEATURES_GP_YEAR = ARTIFACTS / "features_gp_year.pkl"
METRICS = ARTIFACTS / "metrics.json"
PREDICTIONS = ARTIFACTS / "battery_predictions.csv"
PV_METER_FLAGS = BASE.ARTIFACTS / "meter_flags.csv"
PV_OOF = BASE.MODEL_DIR / "oof_predictions.csv"

EPS = 0.01
MIN_EXPORT_DAYS = 10
MIN_DAYS_PER_MONTH = 20
MIN_MONTHS_PER_YEAR = 6
DEFAULT_PI = 0.30
PI_VALUES = (0.20, 0.30, 0.40)

discover_monthly_exports = BASE.discover_monthly_exports

for _path in (ARTIFACTS, MODEL_DIR, FIGURES):
    _path.mkdir(parents=True, exist_ok=True)

