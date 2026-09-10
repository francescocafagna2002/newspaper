# Dataset overview

The repository's data is under [`store/input_data/`](store/input_data/). It contains 46 semicolon-delimited CSV files (about **77.3 GiB** total): 43 monthly electricity-consumption exports and three supporting reference tables.

## Monthly electricity-consumption exports

**Location:** `store/input_data/<year>/<month>/LG_AIM2Hackerdays_kWh_<export timestamp>.csv`

There is one export for each month from **January 2023 through July 2026** (12 files for 2023, 12 for 2024, 12 for 2025, and 7 for 2026). The filename says `kWh`, indicating energy readings; the files are the dominant part of the repository's data volume.

Each export has 101 columns:

- `MP ID` — meter-point ID used by the source exports.
- `OBIS-Code` — measurement/register identifier.
- `Datum` — date.
- `PLZ` — postal code.
- 97 time columns, in 15-minute intervals from `00:15` through `00:00` (the final interval boundary).

The month folders are named in German (for example, `Januar 2024` and `März 2025`). The 2023 exports have one extra directory level: `store/input_data/2023/2023/`.

## Supporting reference tables

| File | Approx. data rows | Purpose and key fields |
| --- | ---: | --- |
| `mpid_zähler_mapping.csv` | 89,993 | Maps `MP ID` to `Zählpunktbezeichnung` (metering-point designation). |
| `Zähler-GP.csv` | 89,910 | Connects `Zählpunktbezeichnung` to `GPartner` and `Anlage` (installation). |
| `HackDays2026 - GIGI.csv` | 1,192 | Customer/site attributes keyed by `GP-Nr`: `PLZ`, `Ort`, `Kanton`, and flags/details for heat pumps, photovoltaic systems, batteries, EV charging, and project dates. |

## How the datasets connect

`MP ID` in a monthly export can be joined to `mpid_zähler_mapping.csv`. From its `Zählpunktbezeichnung`, join to `Zähler-GP.csv`; its `GPartner` corresponds to `GP-Nr` in `HackDays2026 - GIGI.csv`.

```text
monthly export (MP ID)
  -> mpid_zähler_mapping (Zählpunktbezeichnung)
  -> Zähler-GP (GPartner, Anlage)
  -> HackDays2026 - GIGI (GP-Nr and site/technology attributes)
```

## Handling notes

- All files use `;` as the delimiter.
- `HackDays2026 - GIGI.csv` and `Zähler-GP.csv` begin with a UTF-8 byte-order mark; account for it when matching their first column names.
- The monthly exports are individually large (roughly 0.8–3.4 GB), so process them one at a time or stream them rather than loading all files into memory.
