"""Small solar-geometry helpers for canton Aargau.

Smart-meter timestamps are interpreted as Swiss local civil time.  We use
UTC+1 throughout (rather than applying DST) because the resulting one-hour
summer offset is conservative for run detection and is explicit/testable.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

LATITUDE_DEG = 47.4
LONGITUDE_DEG = 8.1
UTC_OFFSET_HOURS = 1.0


def sun_hours_vec(dates) -> tuple[np.ndarray, np.ndarray]:
    """Return sunrise and sunset decimal local hours using NOAA equations."""
    d = pd.DatetimeIndex(pd.to_datetime(dates))
    n = d.dayofyear.to_numpy(float)
    gamma = 2 * np.pi / 365 * (n - 1)
    eqtime = 229.18 * (
        0.000075 + 0.001868 * np.cos(gamma) - 0.032077 * np.sin(gamma)
        - 0.014615 * np.cos(2 * gamma) - 0.040849 * np.sin(2 * gamma)
    )
    decl = (
        0.006918 - 0.399912 * np.cos(gamma) + 0.070257 * np.sin(gamma)
        - 0.006758 * np.cos(2 * gamma) + 0.000907 * np.sin(2 * gamma)
        - 0.002697 * np.cos(3 * gamma) + 0.00148 * np.sin(3 * gamma)
    )
    lat = np.deg2rad(LATITUDE_DEG)
    cos_ha = (
        np.cos(np.deg2rad(90.833)) / (np.cos(lat) * np.cos(decl))
        - np.tan(lat) * np.tan(decl)
    )
    ha_deg = np.rad2deg(np.arccos(np.clip(cos_ha, -1, 1)))
    solar_noon_min = 720 - 4 * LONGITUDE_DEG - eqtime + 60 * UTC_OFFSET_HOURS
    return (solar_noon_min - 4 * ha_deg) / 60, (solar_noon_min + 4 * ha_deg) / 60


def sun_hours(date: pd.Timestamp) -> tuple[float, float]:
    """Return ``(sunrise, sunset)`` in decimal Swiss local civil hours."""
    rise, setting = sun_hours_vec([date])
    return float(rise[0]), float(setting[0])


def main() -> None:
    summer = sun_hours(pd.Timestamp("2025-06-21"))
    winter = sun_hours(pd.Timestamp("2025-12-21"))
    assert 4.0 < summer[0] < 6.5 and 19.0 < summer[1] < 21.5
    assert 7.0 < winter[0] < 9.5 and 15.0 < winter[1] < 17.5
    print(f"summer={summer}, winter={winter}")


if __name__ == "__main__":
    main()

