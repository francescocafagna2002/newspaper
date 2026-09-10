#!/usr/bin/env python3
"""Build the EV-classification customer-week modelling table and audits.

The source monthly exports are intentionally streamed one file at a time.  The
script only retains rows for meter points linked to a GIGI customer and the two
selected OBIS registers, so it does not load the full source collection into
memory.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Iterable
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd


IMPORT_OBIS = "1-1:1.29.0*255"
EXPORT_OBIS = "1-1:2.29.0*255"
REGISTER_COLUMNS = (IMPORT_OBIS, EXPORT_OBIS)
EXPECTED_INTERVALS = 96
WEEK_INTERVALS = 672
MAX_MISSING = 12
MAX_MISSING_RUN = 4
DEFAULT_TIMEZONE = "Europe/Zurich"
FINAL_EXPORT_PATTERN = "LG_AIM2Hackerdays_kWh_*.csv"


@dataclass(frozen=True)
class Inputs:
    gigi: Path
    meter_gp: Path
    mpid_mapping: Path
    monthly_exports: list[Path]


def configure_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )


def normalize_text(value: object) -> str | None:
    if pd.isna(value):
        return None
    text = re.sub(r"\s+", " ", str(value).strip())
    return text.casefold() if text else None


def stripped_columns(frame: pd.DataFrame) -> pd.DataFrame:
    renamed = {column: str(column).replace("\ufeff", "").strip() for column in frame.columns}
    return frame.rename(columns=renamed)


def find_single(data_root: Path, filename: str) -> Path:
    matches = list(data_root.rglob(filename))
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected exactly one {filename!r} below {data_root}, found {len(matches)}")
    return matches[0]


def discover_inputs(data_root: Path) -> Inputs:
    temporary_marker = ".tmp."
    exports = sorted(
        path for path in data_root.rglob(FINAL_EXPORT_PATTERN)
        if temporary_marker not in path.name and path.is_file()
    )
    if not exports:
        raise FileNotFoundError(f"No finalized monthly exports found below {data_root}")
    return Inputs(
        gigi=find_single(data_root, "HackDays2026 - GIGI.csv"),
        meter_gp=find_single(data_root, "Zähler-GP.csv"),
        mpid_mapping=find_single(data_root, "mpid_zähler_mapping.csv"),
        monthly_exports=exports,
    )


def distinct_normalized(values: pd.Series) -> list[str]:
    return sorted({value for value in (normalize_text(item) for item in values) if value is not None})


def modal_value(values: pd.Series) -> tuple[object, bool]:
    """Return a selected source value and a conflict flag.

    A tied mode intentionally has no canonical value.  The raw competing values
    remain in the consolidation audit; no separate tie flag is emitted.
    """
    normalized: list[tuple[str, object]] = []
    for value in values:
        norm = normalize_text(value)
        if norm is not None:
            normalized.append((norm, value))
    if not normalized:
        return None, False
    counts: dict[str, int] = {}
    first_value: dict[str, object] = {}
    for norm, value in normalized:
        counts[norm] = counts.get(norm, 0) + 1
        first_value.setdefault(norm, value)
    maximum = max(counts.values())
    winners = [norm for norm, count in counts.items() if count == maximum]
    if len(winners) > 1:
        return None, len(counts) > 1
    return first_value[winners[0]], len(counts) > 1


def parse_exact_dates(values: pd.Series) -> pd.Series:
    raw = values.astype("string").str.strip()
    exact = raw.str.fullmatch(r"\d{2}\.\d{2}\.\d{4}", na=False)
    parsed = pd.Series(pd.NaT, index=values.index, dtype="datetime64[ns]")
    parsed.loc[exact] = pd.to_datetime(raw.loc[exact], format="%d.%m.%Y", errors="coerce")
    return parsed


def canonicalize_gigi(gigi_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw = stripped_columns(pd.read_csv(gigi_path, sep=";", encoding="utf-8-sig", dtype="string"))
    required = {
        "GP-Nr", "PLZ", "Ort", "Kanton", "WärmePumpe", "PV", "PV-Leistung in kWp",
        "Batterie/Speicher", "Ladestation für Elektrofahrzeuge", "Wärmepumpenboiler",
        "Datum Unterschrift", "geplanter Baustart", "Übergabe", "InBetrieb-Datum",
    }
    missing = required.difference(raw.columns)
    if missing:
        raise ValueError(f"GIGI is missing required columns: {sorted(missing)}")
    raw["gp_nr"] = raw["GP-Nr"].map(normalize_text)
    raw = raw.loc[raw["gp_nr"].notna()].copy()
    asset_columns = {
        "WärmePumpe": "heat_pump_label",
        "PV": "pv_label",
        "Batterie/Speicher": "battery_storage_label",
        "Ladestation für Elektrofahrzeuge": "ev_label",
        "Wärmepumpenboiler": "heat_pump_boiler_label",
    }
    date_columns = {
        "Datum Unterschrift": ("datum_unterschrift_earliest", "min"),
        "geplanter Baustart": ("geplanter_baustart_earliest", "min"),
        "Übergabe": ("uebergabe_latest", "max"),
        "InBetrieb-Datum": ("inbetrieb_datum_latest", "max"),
    }
    records: list[dict[str, object]] = []
    audits: list[dict[str, object]] = []
    for gp_nr, group in raw.groupby("gp_nr", sort=True):
        record: dict[str, object] = {"gp_nr": gp_nr, "gigi_row_count": len(group)}
        audit: dict[str, object] = {"gp_nr": gp_nr, "raw_row_count": len(group)}
        location_conflict = False
        for source, target in (("PLZ", "plz"), ("Ort", "ort"), ("Kanton", "kanton")):
            selected, conflict = modal_value(group[source])
            record[target] = selected
            location_conflict |= conflict
            audit[f"{target}_raw_values"] = json.dumps(distinct_normalized(group[source]), ensure_ascii=False)
        record["gigi_location_conflict"] = location_conflict
        asset_conflict = False
        for source, target in asset_columns.items():
            values = distinct_normalized(group[source])
            conflict = len(values) > 1
            asset_conflict |= conflict
            record[target] = int("x" in values)
            audit[f"{target}_raw_values"] = json.dumps(values, ensure_ascii=False)
            audit[f"{target}_conflict"] = conflict
            if target == "ev_label":
                record["ev_label_conflict"] = conflict
        record["gigi_asset_conflict"] = asset_conflict
        capacity = pd.to_numeric(
            group["PV-Leistung in kWp"].astype("string").str.replace(",", ".", regex=False), errors="coerce"
        )
        record["pv_capacity_kwp"] = capacity.max() if capacity.notna().any() else np.nan
        raw_capacity = distinct_normalized(group["PV-Leistung in kWp"])
        audit["pv_capacity_raw_values"] = json.dumps(raw_capacity, ensure_ascii=False)
        audit["pv_capacity_conflict"] = len(raw_capacity) > 1
        for source, (target, operation) in date_columns.items():
            parsed = parse_exact_dates(group[source])
            record[target] = parsed.min() if operation == "min" else parsed.max()
            raw_values = distinct_normalized(group[source])
            audit[f"{target}_raw_values"] = json.dumps(raw_values, ensure_ascii=False)
            audit[f"{target}_source_conflict"] = len(raw_values) > 1
            audit[f"{target}_unparseable_values"] = json.dumps(
                sorted({normalize_text(value) for value, parsed_value in zip(group[source], parsed) if normalize_text(value) and pd.isna(parsed_value)}),
                ensure_ascii=False,
            )
        record["label_date_ambiguous"] = int(record["ev_label"] == 1)
        record["eligible_for_supervised_training"] = int(not record["ev_label_conflict"])
        record["training_exclusion_reason"] = "ev_label_conflict" if record["ev_label_conflict"] else None
        records.append(record)
        audits.append(audit)
    return pd.DataFrame(records), pd.DataFrame(audits)


def build_mapping(inputs: Inputs, canonical: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    meter_gp = stripped_columns(pd.read_csv(inputs.meter_gp, sep=";", encoding="utf-8-sig", dtype="string"))
    mpid = stripped_columns(pd.read_csv(inputs.mpid_mapping, sep=";", dtype="string"))
    for frame, required, name in ((meter_gp, {"Zählpunktbezeichnung", "GPartner", "Anlage"}, "Zähler-GP"), (mpid, {"MP ID", "Zählpunktbezeichnung"}, "mpid mapping")):
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(f"{name} is missing required columns: {sorted(missing)}")
    meter_gp = meter_gp.rename(columns={"GPartner": "gp_nr", "Zählpunktbezeichnung": "meter_designation", "Anlage": "anlage"})
    meter_gp["gp_nr"] = meter_gp["gp_nr"].map(normalize_text)
    meter_gp["meter_designation"] = meter_gp["meter_designation"].map(normalize_text)
    mpid = mpid.rename(columns={"MP ID": "mp_id", "Zählpunktbezeichnung": "meter_designation"})
    mpid["mp_id"] = mpid["mp_id"].map(normalize_text)
    mpid["meter_designation"] = mpid["meter_designation"].map(normalize_text)
    source_counts = meter_gp.groupby(["gp_nr", "meter_designation"], dropna=False).size().rename("meter_gp_source_row_count").reset_index()
    links = meter_gp.merge(source_counts, on=["gp_nr", "meter_designation"], how="left")
    links = links.drop_duplicates(["gp_nr", "meter_designation"])
    links = links.merge(mpid.drop_duplicates(["mp_id", "meter_designation"]), on="meter_designation", how="left")
    links = links.merge(canonical[["gp_nr"]], on="gp_nr", how="inner")
    links = links.loc[links["mp_id"].notna()].copy()
    links["linked_installation_count"] = links.groupby(["gp_nr", "mp_id"])["anlage"].transform("nunique")
    mp_gp_count = links.groupby("mp_id")["gp_nr"].nunique()
    links["mp_linked_to_multiple_gps"] = links["mp_id"].map(mp_gp_count).gt(1)
    links = links.drop_duplicates(["gp_nr", "mp_id"])
    links["mapping_status"] = np.where(links["mp_linked_to_multiple_gps"], "ambiguous", "linked")
    audit_columns = ["gp_nr", "mp_id", "meter_designation", "linked_installation_count", "mapping_status", "meter_gp_source_row_count", "mp_linked_to_multiple_gps"]
    return links[["gp_nr", "mp_id"]].copy(), links[audit_columns].copy()


def time_columns(columns: Iterable[object]) -> list[str]:
    result = [str(column) for column in columns if re.fullmatch(r"\d{2}:\d{2}", str(column))]
    if len(result) != EXPECTED_INTERVALS or set(result) != {f"{hour:02d}:{minute:02d}" for hour in range(24) for minute in range(0, 60, 15)}:
        raise ValueError(f"Expected exactly {EXPECTED_INTERVALS} quarter-hour columns; found {len(result)}")
    return sorted(result)


def monthly_reader(path: Path, chunksize: int) -> tuple[Iterable[pd.DataFrame], list[str], str]:
    """Return a chunk reader while repairing the one known shifted-header variant."""
    header = stripped_columns(pd.read_csv(path, sep=";", nrows=0))
    intervals = time_columns(header.columns)
    normal = {"MP ID", "OBIS-Code", "Datum", "PLZ", *intervals}
    if normal.issubset(header.columns):
        return pd.read_csv(path, sep=";", dtype="string", chunksize=chunksize), intervals, "standard"
    shifted = {"MP ID", "Datum", "PLZ", *intervals}
    if shifted.issubset(header.columns) and "OBIS-Code" not in header.columns:
        logging.warning("Repairing shifted header without OBIS-Code in %s", path)
        names = ["MP ID", "OBIS-Code", "Datum", "PLZ", *intervals, "_trailing_empty_field"]
        return (
            pd.read_csv(path, sep=";", header=None, skiprows=1, names=names, dtype="string", chunksize=chunksize),
            intervals,
            "shifted_header_repaired",
        )
    raise ValueError(f"Unrecognized monthly schema in {path}: {list(header.columns[:6])}")


def process_month(path: Path, mp_to_gp: dict[str, str], chunksize: int) -> tuple[pd.DataFrame, str]:
    reader, intervals, schema_status = monthly_reader(path, chunksize)
    selected: list[pd.DataFrame] = []
    usecols = ["MP ID", "OBIS-Code", "Datum", *intervals]
    for chunk in reader:
        chunk = stripped_columns(chunk)
        required = set(usecols)
        missing = required.difference(chunk.columns)
        if missing:
            raise ValueError(f"{path} is missing required monthly columns: {sorted(missing)}")
        filtered = chunk.loc[
            chunk["MP ID"].map(normalize_text).isin(mp_to_gp) & chunk["OBIS-Code"].isin(REGISTER_COLUMNS),
            usecols,
        ].copy()
        if not filtered.empty:
            selected.append(filtered)
    if not selected:
        return pd.DataFrame(), schema_status
    frame = pd.concat(selected, ignore_index=True)
    frame["mp_id"] = frame["MP ID"].map(normalize_text)
    frame["gp_nr"] = frame["mp_id"].map(mp_to_gp)
    frame["date"] = pd.to_datetime(frame["Datum"], format="%d.%m.%Y", errors="coerce")
    if frame["date"].isna().any():
        raise ValueError(f"Unparseable dates in {path}")
    intervals = time_columns(frame.columns)
    for column in intervals:
        frame[column] = pd.to_numeric(frame[column].str.replace(",", ".", regex=False), errors="coerce").astype("float32")
    duplicated = frame.duplicated(["mp_id", "date", "OBIS-Code"], keep=False)
    if duplicated.any():
        logging.warning("Skipping %d duplicate register rows in %s", int(duplicated.sum()), path.name)
        frame = frame.loc[~duplicated].copy()
    wide = frame.set_index(["gp_nr", "mp_id", "date", "OBIS-Code"])[intervals].unstack("OBIS-Code")
    wide = wide.reindex(columns=pd.MultiIndex.from_product([intervals, REGISTER_COLUMNS]))
    import_values = wide.xs(IMPORT_OBIS, axis=1, level=1)
    export_values = wide.xs(EXPORT_OBIS, axis=1, level=1)
    net = import_values - export_values
    net.columns = intervals
    net = net.reset_index()
    return net, schema_status


def longest_missing_run(mask: np.ndarray) -> int:
    longest = current = 0
    for missing in mask:
        if missing:
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return longest


def dst_transition_week(week_start: pd.Timestamp, timezone: ZoneInfo) -> bool:
    start = datetime.combine(week_start.date(), datetime.min.time(), tzinfo=timezone)
    end = start + timedelta(days=7)
    return start.utcoffset() != end.utcoffset()


def build_weeks(daily: pd.DataFrame, links: pd.DataFrame, timezone_name: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    intervals = time_columns(daily.columns)
    daily = daily.copy()
    daily["week_start"] = daily["date"] - pd.to_timedelta(daily["date"].dt.weekday, unit="D")
    linked_mps = links.groupby("gp_nr")["mp_id"].agg(lambda values: sorted(set(values))).to_dict()
    quality_rows: list[dict[str, object]] = []
    week_rows: list[dict[str, object]] = []
    timezone = ZoneInfo(timezone_name)
    for (gp_nr, week_start), customer_days in daily.groupby(["gp_nr", "week_start"], sort=True):
        week_start = pd.Timestamp(week_start)
        quality: dict[str, object] = {"gp_nr": gp_nr, "week_start": week_start, "linked_mp_count": len(linked_mps[gp_nr])}
        if dst_transition_week(week_start, timezone):
            quality.update({"observed_timestamp_count": 0, "imputed_timestamp_count": 0, "longest_missing_run": WEEK_INTERVALS, "coverage_ratio": 0.0, "eligible": False, "exclusion_reason": "dst_transition_week"})
            quality_rows.append(quality)
            continue
        day_index = pd.date_range(week_start, periods=7, freq="D")
        values = np.full((7, EXPECTED_INTERVALS), np.nan, dtype="float32")
        for day_position, current_day in enumerate(day_index):
            mps = linked_mps[gp_nr]
            per_meter = customer_days.loc[customer_days["date"] == current_day].set_index("mp_id").reindex(mps)
            if per_meter.empty:
                continue
            values[day_position] = per_meter[intervals].sum(axis=0, min_count=len(mps)).to_numpy(dtype="float32")
        flat = values.reshape(-1)
        missing = np.isnan(flat)
        missing_count = int(missing.sum())
        run = longest_missing_run(missing)
        observed = WEEK_INTERVALS - missing_count
        quality.update({"observed_timestamp_count": observed, "imputed_timestamp_count": 0, "longest_missing_run": run, "coverage_ratio": observed / WEEK_INTERVALS})
        eligible = missing_count <= MAX_MISSING and run <= MAX_MISSING_RUN and (not missing[0]) and (not missing[-1])
        if not eligible:
            if missing_count > MAX_MISSING:
                reason = "too_many_missing_timestamps"
            elif run > MAX_MISSING_RUN:
                reason = "missing_run_too_long"
            else:
                reason = "leading_or_trailing_gap"
            quality.update({"eligible": False, "exclusion_reason": reason})
            quality_rows.append(quality)
            continue
        filled = pd.Series(flat).interpolate(method="linear", limit=MAX_MISSING_RUN, limit_area="inside").to_numpy(dtype="float32")
        if np.isnan(filled).any():
            raise AssertionError("Eligible week still contains missing values after interpolation")
        quality.update({"imputed_timestamp_count": missing_count, "eligible": True, "exclusion_reason": None})
        quality_rows.append(quality)
        row: dict[str, object] = {"gp_nr": gp_nr, "week_start": week_start}
        for day_position, day in enumerate(("mon", "tue", "wed", "thu", "fri", "sat", "sun")):
            for time, value in zip(intervals, filled.reshape(7, EXPECTED_INTERVALS)[day_position]):
                row[f"net_load_{day}_{time.replace(':', '_')}_kwh"] = value
        row.update({key: quality[key] for key in ("linked_mp_count", "observed_timestamp_count", "imputed_timestamp_count", "longest_missing_run", "coverage_ratio")})
        week_rows.append(row)
    return pd.DataFrame(week_rows, columns=["gp_nr", "week_start"] if not week_rows else None), pd.DataFrame(quality_rows)


def write_parquet(frame: pd.DataFrame, path: Path, partition_column: str | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if partition_column and partition_column in frame.columns:
        output = frame.copy()
        output["year"] = pd.to_datetime(output[partition_column]).dt.year
        output.to_parquet(path, index=False, partition_cols=["year"])
    else:
        frame.to_parquet(path, index=False)


def build_summary(canonical: pd.DataFrame, coverage: pd.DataFrame, quality: pd.DataFrame, inputs: Inputs, output: pd.DataFrame) -> pd.DataFrame:
    load_columns = [column for column in output.columns if column.startswith("net_load_")]
    if output.empty:
        pv_negative_fraction = np.nan
        non_pv_negative_fraction = np.nan
    else:
        negative_share = output[load_columns].lt(0).mean(axis=1)
        pv_negative_fraction = negative_share.loc[output["pv_label"].eq(1)].mean()
        non_pv_negative_fraction = negative_share.loc[output["pv_label"].eq(0)].mean()
    return pd.DataFrame([{
        "canonical_customer_count": len(canonical),
        "linked_customer_count": int(coverage["linked_mp_count"].gt(0).sum()),
        "unlinked_customer_count": int(coverage["linked_mp_count"].eq(0).sum()),
        "observed_customer_count": int(coverage["observed_period_count"].gt(0).sum()),
        "eligible_customer_count": int(coverage["eligible_week_count"].gt(0).sum()),
        "attempted_week_count": len(quality),
        "eligible_week_count": int(quality["eligible"].sum()) if not quality.empty else 0,
        "monthly_export_count": len(inputs.monthly_exports),
        "excluded_source": "temporary March 2023 export",
        "import_obis": IMPORT_OBIS,
        "export_obis": EXPORT_OBIS,
        "timezone_assumption": DEFAULT_TIMEZONE,
        "pv_labelled_negative_interval_fraction": pv_negative_fraction,
        "non_pv_labelled_negative_interval_fraction": non_pv_negative_fraction,
    }])


def run(data_root: Path, output_dir: Path, chunksize: int, timezone: str) -> None:
    inputs = discover_inputs(data_root)
    canonical, consolidation_audit = canonicalize_gigi(inputs.gigi)
    links, mapping_audit = build_mapping(inputs, canonical)
    if links["mp_id"].duplicated().any():
        raise ValueError("A mapped MP ID belongs to more than one GIGI customer; cannot safely aggregate")
    mp_to_gp = links.set_index("mp_id")["gp_nr"].to_dict()
    logging.info("Processing %d finalized monthly exports for %d linked meter points", len(inputs.monthly_exports), len(mp_to_gp))
    daily_blocks = []
    monthly_schema_status: dict[str, str] = {}
    for monthly_export in inputs.monthly_exports:
        logging.info("Reading %s", monthly_export)
        month, schema_status = process_month(monthly_export, mp_to_gp, chunksize)
        monthly_schema_status[str(monthly_export.resolve())] = schema_status
        if not month.empty:
            daily_blocks.append(month)
    daily = pd.concat(daily_blocks, ignore_index=True) if daily_blocks else pd.DataFrame()
    if daily.empty:
        raise RuntimeError("No paired register observations were found for linked meter points")
    weeks, quality = build_weeks(daily, links, timezone)
    output = weeks.merge(canonical, on="gp_nr", how="left", validate="many_to_one")
    metadata_columns = [column for column in canonical.columns if column != "gp_nr"]
    load_columns = [
        f"net_load_{day}_{time.replace(':', '_')}_kwh"
        for day in ("mon", "tue", "wed", "thu", "fri", "sat", "sun")
        for time in time_columns(daily.columns)
    ]
    ordered = ["gp_nr", *metadata_columns, "week_start", "linked_mp_count", "observed_timestamp_count", "imputed_timestamp_count", "longest_missing_run", "coverage_ratio", *load_columns]
    output = output.reindex(columns=ordered)
    expected_load_columns = [column for column in output if column.startswith("net_load_")]
    if len(expected_load_columns) != WEEK_INTERVALS or output.duplicated(["gp_nr", "week_start"]).any() or output[expected_load_columns].isna().any(axis=None):
        raise AssertionError("Customer-week output did not satisfy its shape or completeness contract")
    observed_periods = daily.groupby("gp_nr")["date"].nunique().rename("observed_period_count")
    eligible_weeks = quality.loc[quality["eligible"]].groupby("gp_nr").size().rename("eligible_week_count")
    coverage = canonical[["gp_nr"]].merge(links.groupby("gp_nr")["mp_id"].nunique().rename("linked_mp_count"), on="gp_nr", how="left")
    coverage = coverage.merge(observed_periods, on="gp_nr", how="left").merge(eligible_weeks, on="gp_nr", how="left")
    coverage[["linked_mp_count", "observed_period_count", "eligible_week_count"]] = coverage[["linked_mp_count", "observed_period_count", "eligible_week_count"]].fillna(0).astype(int)
    coverage["exclusion_reason"] = np.select([coverage["linked_mp_count"].eq(0), coverage["observed_period_count"].eq(0), coverage["eligible_week_count"].eq(0)], ["unlinked", "no_observations", "no_eligible_weeks"], default=None)
    summary = build_summary(canonical, coverage, quality, inputs, output)
    write_parquet(output, output_dir / "customer_week_table.parquet", "week_start")
    write_parquet(coverage, output_dir / "customer_coverage.parquet")
    write_parquet(mapping_audit, output_dir / "meter_mapping_audit.parquet")
    write_parquet(quality, output_dir / "week_quality_audit.parquet", "week_start")
    write_parquet(consolidation_audit, output_dir / "gigi_consolidation_audit.parquet")
    output_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_dir / "run_summary.csv", index=False)
    metadata = {
        "created_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "data_root": str(data_root.resolve()),
        "finalized_monthly_exports": [str(path.resolve()) for path in inputs.monthly_exports],
        "excluded_source": "temporary March 2023 export",
        "register_convention": {"import_obis": IMPORT_OBIS, "export_obis": EXPORT_OBIS, "formula": "import_kwh - export_kwh", "provenance": "heuristics/pv_negative_load_guided.ipynb"},
        "timezone_assumption": timezone,
        "monthly_schema_status": monthly_schema_status,
        "missing_data_policy": {"max_missing": MAX_MISSING, "max_missing_run": MAX_MISSING_RUN, "internal_gaps_only": True},
    }
    (output_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    logging.info("Wrote %d eligible customer-weeks to %s", len(output), output_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    project_root = Path(__file__).resolve().parents[2]
    parser.add_argument("--data-root", type=Path, default=project_root / "store/input_data", help="Directory containing the three reference CSVs and monthly exports")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "output", help="Directory for Parquet outputs and metadata")
    parser.add_argument("--chunksize", type=int, default=100_000, help="Monthly CSV rows per streaming chunk")
    parser.add_argument("--timezone", default=DEFAULT_TIMEZONE, help="IANA timezone used to identify DST-transition weeks")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    configure_logging(args.verbose)
    run(args.data_root, args.output_dir, args.chunksize, args.timezone)
