"""PV-detection MVP: predict whether a household has a rooftop PV system.

Pipeline (see ../docs/pv_mvp_plan.md and the approved plan):
    build_labels        -> augmented household labels + GP-Nr -> MP ID join
    extract_feedin_labels -> silver positives + export-dominant meter flags
    build_weather       -> MeteoSwiss daily radiation/temperature per PLZ
    build_baserate      -> per-PLZ PV penetration from public registers
    build_features      -> streaming monthly per-meter feature table
    build_features_agg  -> meter -> GP-Nr feature aggregation
    train               -> Positive-Unlabeled model (Elkan-Noto)
    evaluate            -> PU metrics, base-rate check, evidence plots
    score               -> probability per household for the full population
"""
