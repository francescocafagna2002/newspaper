"""Build the WPB slide deck from the artifacts produced by ``wpb.py``.

    python -m models.heat_pump_boiler.wpb_slides

Reads whatever exists in ``models/artifacts`` and skips the rest, so it can
be run at any stage and still produce a usable deck. Requires ``python-pptx``.
"""
from __future__ import annotations

import pandas as pd
from pptx import Presentation
from pptx.util import Inches, Pt

from . import config as C
from . import wpb
from . import wpb_core as W

DECK = C.ARTIFACTS / "wpb_slides.pptx"
INK, ACCENT, MUTED = 0x1F2937, 0x0F766E, 0x6B7280


def _rgb(v):
    from pptx.dml.color import RGBColor
    return RGBColor(v >> 16 & 255, v >> 8 & 255, v & 255)


def _title_slide(prs, title, subtitle):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    tb = s.shapes.add_textbox(Inches(0.7), Inches(2.1), Inches(11.9), Inches(1.4))
    p = tb.text_frame.paragraphs[0]
    p.text = title
    p.font.size, p.font.bold, p.font.color.rgb = Pt(40), True, _rgb(INK)
    tb2 = s.shapes.add_textbox(Inches(0.75), Inches(3.4), Inches(11.9), Inches(1.0))
    p2 = tb2.text_frame.paragraphs[0]
    p2.text = subtitle
    p2.font.size, p2.font.color.rgb = Pt(17), _rgb(MUTED)
    return s


def _bullets(prs, title, lines, note: str | None = None):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    tb = s.shapes.add_textbox(Inches(0.7), Inches(0.5), Inches(11.9), Inches(0.8))
    p = tb.text_frame.paragraphs[0]
    p.text = title
    p.font.size, p.font.bold, p.font.color.rgb = Pt(28), True, _rgb(INK)
    body = s.shapes.add_textbox(Inches(0.8), Inches(1.5), Inches(11.6), Inches(5.0))
    tf = body.text_frame
    tf.word_wrap = True
    for i, line in enumerate(lines):
        para = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        indent = line.startswith("  ")
        para.text = ("– " if indent else "• ") + line.strip()
        para.level = 1 if indent else 0
        para.font.size = Pt(15 if indent else 18)
        para.font.color.rgb = _rgb(MUTED if indent else INK)
        para.space_after = Pt(7)
    if note:
        nb = s.shapes.add_textbox(Inches(0.8), Inches(6.6), Inches(11.6), Inches(0.6))
        np_ = nb.text_frame.paragraphs[0]
        np_.text = note
        np_.font.size, np_.font.italic, np_.font.color.rgb = Pt(12), True, _rgb(ACCENT)
    return s


def _figure(prs, title, png, caption):
    if not png.exists():
        return None
    s = prs.slides.add_slide(prs.slide_layouts[6])
    tb = s.shapes.add_textbox(Inches(0.7), Inches(0.4), Inches(11.9), Inches(0.7))
    p = tb.text_frame.paragraphs[0]
    p.text = title
    p.font.size, p.font.bold, p.font.color.rgb = Pt(26), True, _rgb(INK)
    s.shapes.add_picture(str(png), Inches(1.2), Inches(1.3), width=Inches(10.9))
    cb = s.shapes.add_textbox(Inches(0.8), Inches(6.7), Inches(11.6), Inches(0.6))
    cp = cb.text_frame.paragraphs[0]
    cp.text = caption
    cp.font.size, cp.font.color.rgb = Pt(13), _rgb(MUTED)
    return s


def build() -> str:
    prs = Presentation()
    prs.slide_width, prs.slide_height = Inches(13.333), Inches(7.5)
    fig = wpb.FIGDIR

    _title_slide(
        prs,
        "Finding the Wärmepumpenboiler",
        "A physical, interpretable search for a small daily hot-water load "
        "in four years of 15-minute meter data.",
    )

    _bullets(prs, "The device, and why it is not the heat pump", [
        "A Wärmepumpenboiler heats a 200–300 L hot-water tank, not the house.",
        "  WPB: 0.3–1.0 kW, 3–8 h, ~2 kWh/day, flat across the year",
        "  Resistance Elektroboiler: 2–4 kW, ~1–2 h, ~7 kWh/day",
        "  Space heat pump: many cycles/day, tracks heating degree-days",
        "Same hot water, different power — the ratio is roughly the COP.",
        "Why it matters: ~2 kWh/day of shiftable load on a tank that is "
        "already a thermal store.",
    ], "The amplitude split is physics, not a fitted threshold.")

    _bullets(prs, "The constraint that shaped everything", [
        "10 households in GIGI are flagged Wärmepumpenboiler. 4 have meter data.",
        "  '-' in the column means 'row is about another asset', never 'no WPB'",
        "  → zero confirmed negatives, so no supervised classifier",
        "So: a rule-based physical detector, validated without labels.",
    ], "Reported as a sensitivity study, not an accuracy number.")

    _bullets(prs, "Method — the p10 'always-on' envelope", [
        "Per meter-month, take per-slot quantiles across all days.",
        "Run detection on the 10th percentile, not the mean.",
        "  a load must appear on ≥90% of days to lift p10",
        "  → once-a-day timer regularity is enforced structurally",
        "  → an EV charged a few nights a week cannot produce a plateau",
        "Plateau = contiguous run ≥0.25 kW above the quietest rolling hour, "
        "wrapping midnight.",
        "Confirm in BOTH a summer and a winter month.",
        "  dehumidifier, floor heating, part-load heat pump: all seasonal",
        "  a WPB tracks cold-water inlet temperature, not degree-days",
    ])

    _figure(prs, "Evidence: the labelled households",
            fig / "wpb_labelled_profiles.png",
            "Median day profile (solid) with the p10 always-on envelope "
            "(dashed). Dotted line = the 0.125 kWh/slot a 0.5 kW WPB adds.")
    _figure(prs, "Go/no-go: can a 0.5 kW plateau be seen at all?",
            fig / "wpb_noise_floor.png",
            "Night-load noise floor per meter against the step a WPB makes. "
            "If the floor sits above the line, nothing else works.")
    _figure(prs, "Is the population bimodal?",
            fig / "wpb_amplitude_hist.png",
            "Plateau amplitude across all meters. Two clusters near 0.5 kW "
            "and 3 kW would be the WPB / resistance split showing itself.")
    _figure(prs, "Validation without labels: counterfactual retrofit",
            fig / "wpb_detectability.png",
            "Take a real resistance boiler's daily event, hold its energy "
            "fixed, trade power for duration — the Elektroboiler→WPB swap — "
            "add it to an unlabelled household, and see if we find it.")

    if wpb.PREDICTIONS.exists():
        df = pd.read_csv(wpb.PREDICTIONS, low_memory=False)
        hi = df[df["wpb_score"] >= 0.8]
        lines = [
            f"{len(df):,} meters scored; {len(hi):,} flagged at score ≥ 0.8 "
            f"({len(hi)/max(len(df),1):.1%}).",
        ]
        if "amplitude_kw" in hi and len(hi):
            amp = pd.to_numeric(hi["amplitude_kw"], errors="coerce").dropna()
            kwh = pd.to_numeric(hi["daily_kwh"], errors="coerce").dropna()
            if len(amp):
                lines.append(f"  median amplitude {amp.median():.2f} kW "
                             f"(band {W.WPB_KW_MIN}–{W.WPB_KW_MAX} kW)")
            if len(kwh):
                lines.append(f"  median {kwh.median():.2f} kWh/day "
                             f"(a WPB tank is ~1.5–2.5)")
        lab = df[df.get("is_labelled_wpb", False) == True]  # noqa: E712
        if len(lab):
            lines.append(f"Labelled WPB households score "
                         f"{lab['wpb_score'].min():.2f}–"
                         f"{lab['wpb_score'].max():.2f}.")
        lines += [
            "Every row carries its evidence: amplitude, duration, start time, "
            "kWh/day, and why.",
        ]
        household_path = C.ARTIFACTS / "wpb_household_predictions.csv"
        if household_path.exists():
            hh = pd.read_csv(household_path, low_memory=False)
            lines.insert(0, f"{int(hh['wpb_flag'].sum()):,} of {len(hh):,} mapped "
                         "households flagged in the latest pair.")
        _bullets(prs, "Population result", lines,
                 "Inspectable evidence, not a black-box label.")

    corr_path = C.ARTIFACTS / "wpb_gwr_correlations.csv"
    if corr_path.exists():
        corr = pd.read_csv(corr_path)
        row = corr[(corr.min_meters == 30) & (corr.reference == "gwr_hp_rate")].iloc[0]
        _bullets(prs, "External comparison does not confirm the flags", [
            f"Across {int(row.n_postcodes)} postcodes with ≥30 meters, "
            f"Spearman ρ = {row.rho:.2f} (p = {row.pvalue:.2f}).",
            "The GWR field covers all hot-water heat pumps, including combined "
            "space-heating systems; it is a broader ecological comparator.",
            "The weak association is inconclusive and provides no accuracy claim.",
        ])

    _bullets(prs, "What we are not claiming", [
        "No accuracy figure — 4 positives and no confirmed negatives.",
        "A PV-controlled WPB can run at midday and be invisible in grid import.",
        "  GIGI is ~9:1 PV-skewed, so this creates selection bias",
        "None of the four labelled meters passes the year-round threshold.",
        "Detected duration is a lower bound: start jitter erodes the p10 core.",
        "Next: 4-year change detection finds the install date; the "
        "Elektroboiler→WPB swap is the most legible fingerprint in the data.",
    ])

    C.ARTIFACTS.mkdir(parents=True, exist_ok=True)
    prs.save(DECK)
    return str(DECK)


if __name__ == "__main__":
    print("wrote", build())
