"""Space-heating heat-pump (`has_Waermepumpe`) classifier.

Distinct from `models.heat_pump_boiler`, which detects hot-water heat-pump
boilers (WPBs) via a physics-based plateau rule. This package is a supervised
classifier on the GIGI-labelled property-sample table, following the
`ev_classification_total` protocol (group-safe split, sealed test, leaderboard).
Its one addition over that generic recipe is joining the WPB detector's score
and the dedicated-heat-pump-meter aux features as engineered inputs.

    python -m models.heat_pump.build_features
    python -m models.heat_pump.run_classification
    python -m models.heat_pump.slides
"""
