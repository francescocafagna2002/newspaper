"""Build the SolarPrint slide deck.

    python -m newspaper.models.pv.modeling.figures   # (re)generate the PNGs first
    python -m newspaper.models.pv.modeling.slides

Reads artifacts/figures/*.png + artifacts/metrics.json + model/meta.json.
Heuristic-baseline numbers are supplied by the user -> edit HEURISTIC below.
Output: ../../docs/pv_model_slides.pptx
"""
from __future__ import annotations

import json

from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN

from .. import config as C

# --- baseline the user will fill in by hand -----------------------------------
HEURISTIC = {"name": "Heuristic rule", "roc_auc": None,
             "recall_known": None, "flag_rate": None, "note": "supplied separately"}
# -----------------------------------------------------------------------------

FIG = C.ARTIFACTS / "figures"
INK = RGBColor(0x11, 0x11, 0x1A)
ACCENT = RGBColor(0x2A, 0x78, 0xD6)
GREY = RGBColor(0x52, 0x51, 0x4E)
W, H = Inches(13.333), Inches(7.5)


def _title(s, text, sub=None):
    tb = s.shapes.add_textbox(Inches(0.55), Inches(0.32), Inches(12.2), Inches(1.0))
    tf = tb.text_frame; tf.word_wrap = True
    r = tf.paragraphs[0].add_run(); r.text = text
    r.font.size = Pt(26); r.font.bold = True; r.font.color.rgb = INK
    if sub:
        p = tf.add_paragraph(); r = p.add_run(); r.text = sub
        r.font.size = Pt(12); r.font.color.rgb = GREY


def _pic(s, name, l, t, w=None, h=None):
    kw = {}
    if w: kw["width"] = Inches(w)
    if h: kw["height"] = Inches(h)
    s.shapes.add_picture(str(FIG / name), Inches(l), Inches(t), **kw)


def _bullets(s, l, t, w, h, items, *, size=12, head=None):
    tb = s.shapes.add_textbox(Inches(l), Inches(t), Inches(w), Inches(h))
    tf = tb.text_frame; tf.word_wrap = True
    first = True
    if head:
        r = tf.paragraphs[0].add_run(); r.text = head
        r.font.bold = True; r.font.size = Pt(size + 1); r.font.color.rgb = ACCENT
        first = False
    for it in items:
        p = tf.paragraphs[0] if first else tf.add_paragraph(); first = False
        r = p.add_run(); r.text = "•  " + it
        r.font.size = Pt(size); r.font.color.rgb = INK
    return tb


def _sources(s):
    tb = s.shapes.add_textbox(Inches(0.55), Inches(6.35), Inches(12.2), Inches(1.05))
    tf = tb.text_frame; tf.word_wrap = True
    r = tf.paragraphs[0].add_run(); r.text = "Sources"
    r.font.size = Pt(10); r.font.bold = True; r.font.color.rgb = ACCENT
    for line in [
        "Duck-curve midday depression — NREL, Denholm et al. 2015   ·   "
        "Weather–solar coupling — SunDance, Chen & Irwin, ACM e-Energy 2017   ·   "
        "Net-load + weather-status ML detection — Electricity Journal 2022",
        "Weather — MeteoSwiss SwissMetNet open data   ·   Base rate — BFE ElPA register"
        "   ·   Method — challenge label-first approach",
    ]:
        p = tf.add_paragraph(); r = p.add_run(); r.text = line
        r.font.size = Pt(9); r.font.color.rgb = GREY


# --------------------------------------------------------------------------- #
def s_architecture(prs):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    _title(s, "SolarPrint — architecture",
           "Per-household P(rooftop PV) from 15-min grid-import readings (OBIS 1.29 only)")
    _pic(s, "fig_architecture.png", 0.9, 1.25, w=11.6)
    _bullets(s, 0.6, 6.55, 12.2, 0.9, [
        "Positive–Unlabeled (no confirmed non-PV households) · GBDT · 5-fold CV · "
        "Elkan–Noto correction · calibrated to the 12 % ElPA base rate",
    ], size=11.5)


def s_features(prs):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    _title(s, "SolarPrint — features",
           "23 features from the grid-import load shape; feed-in register used offline only")
    _pic(s, "fig_feature_sep.png", 0.3, 1.4, w=7.1)
    _pic(s, "fig_importance.png", 7.55, 1.6, w=5.35)
    _bullets(s, 7.6, 4.75, 5.4, 1.6, [
        "Midday depression / duck curve",
        "Weather–solar coupling (MeteoSwiss radiation)",
        "Seasonality · load level · night base load",
        "Year-on-year change-point (new installs)",
    ], head="Feature groups", size=10.5)
    _sources(s)


def s_signal(prs):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    _title(s, "SolarPrint — the dominant signal",
           "When PV covers the load, net grid-import falls to ~zero during the day")
    _pic(s, "fig_top_feature.png", 1.3, 1.5, w=8.2)
    _bullets(s, 9.7, 1.9, 3.3, 4.5, [
        "PV homes: most summer daytime intervals near zero net import",
        "Rest: daytime import stays well above zero",
        "This one feature, trained the same way, gives ROC-AUC 0.90 "
        "(recall 0.87) — the other 22 add ~2–3 points and sharper probabilities",
    ], size=11)


def _cell(t, r, c, txt, *, bold=False, size=11, color=INK, fill=None, align=PP_ALIGN.LEFT):
    cell = t.cell(r, c); cell.text = str(txt)
    p = cell.text_frame.paragraphs[0]; p.alignment = align
    p.runs[0].font.size = Pt(size); p.runs[0].font.bold = bold
    p.runs[0].font.color.rgb = color
    if fill:
        cell.fill.solid(); cell.fill.fore_color.rgb = fill


def s_results(prs, metrics):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    _title(s, "SolarPrint — results vs baselines",
           "5-fold out-of-fold; held-out GIGI positives as ground truth. "
           "PU-adjusted: ranking reliable, absolute calibration approximate.")
    _pic(s, "fig_results.png", 0.4, 1.5, w=7.6)

    m_roc = metrics["roc_auc_gigi_vs_rest"]; m_rec = metrics["recall@0.5_gigi_positives"]
    m_flag = metrics["population_flag_rate@0.5"]
    fmt = lambda x: "—" if x is None else f"{x:.2f}"
    rows = [
        ("Approach", "ROC-AUC", "Recall@0.5", "Flag-rate"),
        ("Random assignment", "0.50", "0.12", "0.12"),
        (HEURISTIC["name"], fmt(HEURISTIC["roc_auc"]), fmt(HEURISTIC["recall_known"]),
         fmt(HEURISTIC["flag_rate"])),
        ("1 feature (daytime-zero)", "0.90", "0.87", "0.08"),
        ("Full model (23 feat.)", f"{m_roc:.2f}", f"{m_rec:.2f}", f"{m_flag:.2f}"),
    ]
    tbl = s.shapes.add_table(len(rows), 4, Inches(8.2), Inches(1.7),
                             Inches(4.8), Inches(2.4)).table
    tbl.columns[0].width = Inches(2.4)
    for c in range(1, 4):
        tbl.columns[c].width = Inches(0.8)
    for j, htxt in enumerate(rows[0]):
        _cell(tbl, 0, j, htxt, bold=True, size=10,
              color=RGBColor(0xFF, 0xFF, 0xFF), fill=INK, align=PP_ALIGN.CENTER)
    for i, row in enumerate(rows[1:], 1):
        hero = row[0].startswith("Full model")
        for j, v in enumerate(row):
            _cell(tbl, i, j, v, bold=hero, size=10,
                  color=ACCENT if hero else INK,
                  fill=RGBColor(0xEC, 0xF3, 0xFB) if hero else None,
                  align=PP_ALIGN.LEFT if j == 0 else PP_ALIGN.CENTER)

    _bullets(s, 8.2, 4.4, 4.8, 2.4, [
        f"Flags {m_flag:.0%} of the population (ElPA base rate ≈ 12 %) and "
        f"catches {m_rec:.0%} of held-out known PV homes.",
        "Weak spots: geographic agreement with the PV register modest "
        "(ρ≈0.1); change-point feature does not track install dates yet.",
    ], size=10.5)


def main():
    metrics = json.loads((C.ARTIFACTS / "metrics.json").read_text())
    prs = Presentation()
    prs.slide_width = W; prs.slide_height = H
    s_architecture(prs)
    s_features(prs)
    s_signal(prs)
    s_results(prs, metrics)
    out = C.NEWSPAPER / "docs" / "pv_model_slides.pptx"
    prs.save(out)
    print(f"wrote {out}  ({len(prs.slides.__iter__.__self__._sldIdLst)} slides)")
    if HEURISTIC["roc_auc"] is None:
        print("  NOTE: heuristic row is a placeholder — set HEURISTIC in modeling/slides.py")


if __name__ == "__main__":
    main()
