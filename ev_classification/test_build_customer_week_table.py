import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd


MODULE_PATH = Path(__file__).with_name("build_customer_week_table.py")
SPEC = importlib.util.spec_from_file_location("customer_week_builder", MODULE_PATH)
builder = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = builder
SPEC.loader.exec_module(builder)


class CustomerWeekTableTest(unittest.TestCase):
    def test_repairs_a_shifted_monthly_header(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "shifted.csv"
            times = [f"{hour:02d}:{minute:02d}" for hour in range(24) for minute in range(0, 60, 15)]
            path.write_text(
                ";".join(["MP ID", "Datum", "PLZ", *times]) + ";\n"
                + ";".join(["mp-1", builder.IMPORT_OBIS, "01.01.2024", "5000", *("0.0" for _ in times)]) + ";\n",
                encoding="utf-8",
            )
            reader, intervals, status = builder.monthly_reader(path, chunksize=1)
            row = next(iter(reader)).iloc[0]
            reader.close()
            self.assertEqual(status, "shifted_header_repaired")
            self.assertEqual(len(intervals), 96)
            self.assertEqual(row["MP ID"], "mp-1")
            self.assertEqual(row["OBIS-Code"], builder.IMPORT_OBIS)
            self.assertEqual(row["Datum"], "01.01.2024")

    def test_builds_an_imputed_week_and_all_audits(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory) / "input_data"
            root.mkdir()
            pd.DataFrame([
                {"GP-Nr": "customer-1", "PLZ": "5000", "Ort": "Aarau", "Kanton": "AG", "WärmePumpe": "-", " PV": "-", "PV-Leistung in kWp ": "", "Batterie/Speicher": "-", "Ladestation für Elektrofahrzeuge": "x", "Wärmepumpenboiler": "-", "Datum Unterschrift": "01.01.2024", "geplanter Baustart": "invalid", "Übergabe": "", "InBetrieb-Datum": ""},
                {"GP-Nr": "customer-1", "PLZ": "5000", "Ort": "Aarau", "Kanton": "AG", "WärmePumpe": "-", " PV": "-", "PV-Leistung in kWp ": "", "Batterie/Speicher": "-", "Ladestation für Elektrofahrzeuge": "-", "Wärmepumpenboiler": "-", "Datum Unterschrift": "01.01.2024", "geplanter Baustart": "invalid", "Übergabe": "", "InBetrieb-Datum": ""},
            ]).to_csv(root / "HackDays2026 - GIGI.csv", sep=";", index=False, encoding="utf-8-sig")
            pd.DataFrame([{"Zählpunktbezeichnung": "meter-1", "GPartner": "customer-1", "Anlage": "install-1"}]).to_csv(root / "Zähler-GP.csv", sep=";", index=False, encoding="utf-8-sig")
            pd.DataFrame([{"MP ID": "mp-1", "Zählpunktbezeichnung": "meter-1"}]).to_csv(root / "mpid_zähler_mapping.csv", sep=";", index=False)
            times = [f"{hour:02d}:{minute:02d}" for hour in range(24) for minute in range(0, 60, 15)]
            source_times = times[1:] + times[:1]
            rows = []
            for current_day in pd.date_range("2024-01-01", periods=7, freq="D"):
                for obis, value in ((builder.IMPORT_OBIS, "2.0"), (builder.EXPORT_OBIS, "0.5")):
                    row = {"MP ID": "mp-1", "OBIS-Code": obis, "Datum": current_day.strftime("%d.%m.%Y"), "PLZ": "5000", **{time: value for time in source_times}}
                    if current_day.day == 3 and obis == builder.IMPORT_OBIS:
                        row["12:00"] = ""
                    rows.append(row)
            month_dir = root / "2024" / "Januar 2024"
            month_dir.mkdir(parents=True)
            pd.DataFrame(rows).to_csv(month_dir / "LG_AIM2Hackerdays_kWh_20240101_000000.csv", sep=";", index=False)
            output = Path(temporary_directory) / "output"
            builder.run(root, output, chunksize=5, timezone="Europe/Zurich")
            table = pd.read_parquet(output / "customer_week_table.parquet")
            quality = pd.read_parquet(output / "week_quality_audit.parquet")
            coverage = pd.read_parquet(output / "customer_coverage.parquet")
            summary = pd.read_csv(output / "run_summary.csv")
            self.assertEqual(len(table), 1)
            self.assertEqual(sum(column.startswith("net_load_") for column in table.columns), 672)
            self.assertEqual(table.loc[0, "imputed_timestamp_count"], 1)
            self.assertFalse(table.loc[0, "eligible_for_supervised_training"])
            self.assertEqual(table.loc[0, "training_exclusion_reason"], "ev_label_conflict")
            self.assertAlmostEqual(table.loc[0, "net_load_wed_12_00_kwh"], 1.5)
            self.assertTrue(quality.loc[0, "eligible"])
            self.assertEqual(coverage.loc[0, "eligible_week_count"], 1)
            self.assertIn("pv_labelled_negative_interval_fraction", summary.columns)


if __name__ == "__main__":
    unittest.main()
