"""Build the one-slide heat-pump deck: top feature, confusion matrix, accuracy.

    python -m models.heat_pump.run_classification   # produces the inputs
    python -m models.heat_pump.slides

Reads artifacts/final_freeze.json + artifacts/feature_importance.csv (and
artifacts/population_gwr_comparison.json for the base-rate footnote, if built).
Output: artifacts/heat_pump_slide.pptx
"""
from __future__ import annotations

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.util import Inches, Pt

from . import config as C

INK = RGBColor(0x11, 0x11, 0x1A)
ACCENT = RGBColor(0x2A, 0x78, 0xD6)
GREY = RGBColor(0x52, 0x51, 0x4E)
MUTED = RGBColor(0xD8, 0xDE, 0xE6)
W, H = Inches(13.333), Inches(7.5)
N_FEATURES = 8


def _importance_figure(importances: pd.DataFrame, top_feature: str) -> str:
    top = importances.head(N_FEATURES).iloc[::-1]
    colors = ["#2A78D6" if f == top_feature else "#B9C4D2" for f in top.feature]
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.barh(top.feature, top.importance, xerr=top.importance_std,
            color=colors, error_kw={"ecolor": "#8A939F", "elinewidth": 0.8, "capsize": 2})
    ax.set_xlabel("permutation importance (accuracy points, out-of-sample)")
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=9)
    fig.tight_layout()
    path = C.FIGURES / "fig_importance.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return str(path)


def _title(slide, text, sub=None):
    box = slide.shapes.add_textbox(Inches(0.55), Inches(0.3), Inches(12.2), Inches(1.0))
    frame = box.text_frame
    frame.word_wrap = True
    run = frame.paragraphs[0].add_run()
    run.text = text
    run.font.size = Pt(26)
    run.font.bold = True
    run.font.color.rgb = INK
    if sub:
        para = frame.add_paragraph()
        run = para.add_run()
        run.text = sub
        run.font.size = Pt(12)
        run.font.color.rgb = GREY


def _cell(table, r, c, text, *, bold=False, size=12, color=INK, fill=None, align=PP_ALIGN.CENTER):
    cell = table.cell(r, c)
    cell.text = str(text)
    para = cell.text_frame.paragraphs[0]
    para.alignment = align
    for run in para.runs:  # an empty cell has no run to style
        run.font.size = Pt(size)
        run.font.bold = bold
        run.font.color.rgb = color
    if fill:
        cell.fill.solid()
        cell.fill.fore_color.rgb = fill


def _confusion_table(slide, cm, left, top):
    (tn, fp), (fn, tp) = cm
    table = slide.shapes.add_table(3, 3, Inches(left), Inches(top),
                                   Inches(4.6), Inches(2.0)).table
    table.columns[0].width = Inches(1.6)
    table.columns[1].width = Inches(1.5)
    table.columns[2].width = Inches(1.5)
    _cell(table, 0, 0, "", fill=INK)
    _cell(table, 0, 1, "predicted: no HP", bold=True, size=10,
          color=RGBColor(0xFF, 0xFF, 0xFF), fill=INK)
    _cell(table, 0, 2, "predicted: HP", bold=True, size=10,
          color=RGBColor(0xFF, 0xFF, 0xFF), fill=INK)
    _cell(table, 1, 0, "actual: no HP", bold=True, size=10, fill=MUTED, align=PP_ALIGN.LEFT)
    _cell(table, 2, 0, "actual: HP", bold=True, size=10, fill=MUTED, align=PP_ALIGN.LEFT)
    _cell(table, 1, 1, f"{tn}  (TN)", bold=True, size=13, color=ACCENT)
    _cell(table, 1, 2, f"{fp}  (FP)", size=13)
    _cell(table, 2, 1, f"{fn}  (FN)", size=13)
    _cell(table, 2, 2, f"{tp}  (TP)", bold=True, size=13, color=ACCENT)


def _stat(slide, left, top, value, label, *, big=40):
    box = slide.shapes.add_textbox(Inches(left), Inches(top), Inches(2.3), Inches(1.1))
    frame = box.text_frame
    frame.word_wrap = True
    run = frame.paragraphs[0].add_run()
    run.text = value
    run.font.size = Pt(big)
    run.font.bold = True
    run.font.color.rgb = ACCENT
    para = frame.add_paragraph()
    run = para.add_run()
    run.text = label
    run.font.size = Pt(10)
    run.font.color.rgb = GREY


def _lines(slide, left, top, width, height, lines, *, size=11, head=None, bullet=True):
    box = slide.shapes.add_textbox(Inches(left), Inches(top), Inches(width), Inches(height))
    frame = box.text_frame
    frame.word_wrap = True
    first = True
    if head:
        run = frame.paragraphs[0].add_run()
        run.text = head
        run.font.bold = True
        run.font.size = Pt(size + 1)
        run.font.color.rgb = ACCENT
        first = False
    for line in lines:
        para = frame.paragraphs[0] if first else frame.add_paragraph()
        first = False
        run = para.add_run()
        run.text = ("•  " if bullet else "") + line
        run.font.size = Pt(size)
        run.font.color.rgb = INK


def main() -> None:
    freeze = json.loads((C.ARTIFACTS / "final_freeze.json").read_text())
    importances = pd.read_csv(C.ARTIFACTS / "feature_importance.csv")
    test = freeze["test_metrics"]
    validation = freeze["validation_metrics"]
    top_feature = freeze["top_feature"]
    top_row = importances.iloc[0]
    figure = _importance_figure(importances, top_feature)

    baseline = 1 - test["prevalence"]
    prs = Presentation()
    prs.slide_width, prs.slide_height = W, H
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    _title(slide, "Heat-pump classifier — most important feature and performance",
           f"{freeze['champion']} · held-out test, {test['n']} properties "
           f"({test['positives']} heat-pump-labelled) · threshold {freeze['threshold']:.2f} "
           f"frozen on validation")

    slide.shapes.add_picture(figure, Inches(0.55), Inches(1.5), width=Inches(6.2))
    _lines(slide, 0.6, 5.62, 6.4, 0.4, [
        f"Top feature: {top_feature} — permuting it costs "
        f"{top_row['importance']:.3f} accuracy points (±{top_row['importance_std']:.3f}), "
        f"more than any seasonality or meter feature.",
    ], size=10, bullet=False)

    _lines(slide, 0.6, 6.1, 6.6, 1.2, [
        "Temperature coupling (daily import vs heating-degree-days) is not "
        "implemented: it needs per-household daily series, and neither "
        "train_property_sequences.csv nor load_property_daily.csv exists in this checkout.",
        "The WPB boiler-detector score contributes little — it flags only 2 of "
        "342 households.",
        "Both are documented in models/heat_pump/README.md, not papered over.",
    ], size=8.5, head="Known gaps vs the plan")

    _stat(slide, 7.6, 1.6, f"{test['accuracy']:.1%}", "accuracy on the sealed test set")
    _stat(slide, 10.4, 1.6, f"{baseline:.1%}",
          "majority-class baseline (predict 'no heat pump')")

    _lines(slide, 7.65, 2.95, 5.2, 0.4, ["Confusion matrix — held-out test"],
           size=12, bullet=False)
    _confusion_table(slide, test["confusion_matrix"], 7.6, 3.35)

    _lines(slide, 7.6, 5.5, 5.3, 0.8, [
        f"Validation accuracy {validation['accuracy']:.1%}; the model recovers "
        f"{test['confusion_matrix'][1][1]} of {test['positives']} labelled heat pumps "
        f"at {test['confusion_matrix'][0][1]} false positives.",
        "Label 0 is subsidy-register-unlabelled, not verified heat-pump-absent, "
        "so accuracy is a floor estimate.",
    ], size=8.5)

    comparison = C.ARTIFACTS / "population_gwr_comparison.json"
    if comparison.exists():
        pop = json.loads(comparison.read_text())["comparison"]
        matched = pop["per_plz_spearman_at_gwr_matched_rate"][-1]
        _lines(slide, 7.6, 6.45, 5.3, 0.8, [
            f"At the GWR rate ({pop['gwr_hp_rate_all_buildings']:.0%} of AG residential "
            f"buildings) the score ranks postcodes with Spearman ρ={matched['rho']:.2f} "
            f"(n={matched['n_postcodes']}), but the accuracy-optimal threshold flags "
            f"{pop['predicted_over_expected']:.1f}× too many households — rank is usable, "
            f"absolute calibration is not.",
        ], size=8.5, head="Population check vs GWR", bullet=False)

    out = C.ARTIFACTS / "heat_pump_slide.pptx"
    prs.save(out)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
