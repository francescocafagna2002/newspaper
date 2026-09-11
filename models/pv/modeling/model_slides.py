"""Build one single-slide PPTX per model (takeaways + confusion matrix).

    python -m newspaper.models.pv.modeling.figures    # (re)generate the PNGs first
    python -m newspaper.models.pv.modeling.model_slides

Reuses the slide-building functions in slides.py; writes one file per model to
../../../docs/model_slides/{pv,ev,battery,wpb}.pptx.
"""
from __future__ import annotations

import json

from pptx import Presentation

from .. import config as C
from . import slides as S

MODELS = ("pv", "ev", "battery", "wpb")


def _new_deck():
    prs = Presentation()
    prs.slide_width = S.W; prs.slide_height = S.H
    return prs


def main():
    out_dir = C.NEWSPAPER / "docs" / "model_slides"
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics = json.loads((C.ARTIFACTS / "metrics.json").read_text())

    pv = _new_deck(); S.s_pv_summary(pv, metrics); pv.save(out_dir / "pv.pptx")
    ev = _new_deck(); S.s_ev_results(ev); ev.save(out_dir / "ev.pptx")
    battery = _new_deck(); S.s_battery_status(battery); battery.save(out_dir / "battery.pptx")
    wpb = _new_deck(); S.s_wpb_status(wpb); wpb.save(out_dir / "wpb.pptx")
    print(f"wrote {len(MODELS)} single-slide decks -> {out_dir}")

    merged = _new_deck()
    S.s_pv_summary(merged, metrics)
    S.s_ev_results(merged)
    S.s_battery_status(merged)
    S.s_wpb_status(merged)
    merged_path = out_dir / "all_models.pptx"
    merged.save(merged_path)
    print(f"wrote merged deck -> {merged_path}")


if __name__ == "__main__":
    main()
