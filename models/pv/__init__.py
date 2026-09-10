"""SolarPrint - household rooftop-PV classifier.

Pipeline stages:
    prep/      labels, silver labels, weather, base rate
    features/  streaming monthly extraction -> per-household aggregation
    modeling/  Positive-Unlabeled train -> evaluate -> score -> slides

Run everything: `python -m newspaper.models.pv.pipeline`
"""
