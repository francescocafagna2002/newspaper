"""Small IO helpers shared across the pipeline."""
from __future__ import annotations

import pandas as pd

from . import config as C


# --------------------------------------------------------------------------- #
# Reference tables
# --------------------------------------------------------------------------- #
def read_gigi(annotated: bool = True) -> pd.DataFrame:
    """Load the GIGI label file with tidy column names.

    The source has cosmetic whitespace in headers (``' PV'``,
    ``'PV-Leistung in kWp '``) and a UTF-8 BOM; we normalise those here.
    """
    path = C.GIGI_ANNOTATED if annotated and C.GIGI_ANNOTATED.exists() else C.GIGI_RAW
    df = pd.read_csv(path, sep=C.CSV_SEP, encoding=C.CSV_ENCODING, dtype=str)
    df.columns = [c.strip() for c in df.columns]
    rename = {
        "GP-Nr": "gp_nr",
        "PLZ": "plz",
        "Ort": "ort",
        "Kanton": "kanton",
        "WärmePumpe": "heat_pump",
        "PV": "pv",
        "PV-Leistung in kWp": "pv_kwp",
        "Batterie/Speicher": "battery",
        "Ladestation für Elektrofahrzeuge": "ev_charger",
        "Wärmepumpenboiler": "hp_boiler",
        "Datum Unterschrift": "date_signature",
        "geplanter Baustart": "planned_start",
        "Übergabe": "date_handover",
        "InBetrieb-Datum": "date_commissioned",
        "InBetrieb-Datum bezieht sich auf": "commissioned_refers_to",
    }
    df = df.rename(columns=rename)
    for col in df.columns:
        df[col] = df[col].str.strip()
    return df


def read_mpid_mapping() -> pd.DataFrame:
    """MP ID <-> Zählpunktbezeichnung."""
    df = pd.read_csv(C.MPID_MAPPING, sep=C.CSV_SEP, encoding=C.CSV_ENCODING, dtype=str)
    df.columns = [c.strip() for c in df.columns]
    return df.rename(columns={"MP ID": "mp_id", "Zählpunktbezeichnung": "zpb"})


def read_zaehler_gp() -> pd.DataFrame:
    """Zählpunktbezeichnung -> GPartner (== GP-Nr) + Anlage."""
    df = pd.read_csv(C.ZAEHLER_GP, sep=C.CSV_SEP, encoding=C.CSV_ENCODING, dtype=str)
    df.columns = [c.strip() for c in df.columns]
    return df.rename(
        columns={"Zählpunktbezeichnung": "zpb", "GPartner": "gp_nr", "Anlage": "anlage"}
    )


def gp_to_mpid() -> pd.DataFrame:
    """Full join chain: one row per (gp_nr, zpb, anlage, mp_id)."""
    zg = read_zaehler_gp()
    mp = read_mpid_mapping()
    return zg.merge(mp, on="zpb", how="left")


# --------------------------------------------------------------------------- #
# Feature-table persistence (no pyarrow in this env -> pickle)
# --------------------------------------------------------------------------- #
def save_table(df: pd.DataFrame, path) -> None:
    path = str(path)
    if path.endswith(".parquet"):
        df.to_parquet(path)
    else:
        df.to_pickle(path)


def load_table(path) -> pd.DataFrame:
    path = str(path)
    if path.endswith(".parquet"):
        return pd.read_parquet(path)
    return pd.read_pickle(path)
