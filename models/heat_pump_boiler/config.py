"""Reuse the canonical data configuration, with separate WPB artifacts."""
from ..pv.config import *  # noqa: F401,F403
ARTIFACTS = NEWSPAPER / "models" / "heat_pump_boiler" / "artifacts"
ARTIFACTS.mkdir(parents=True, exist_ok=True)
