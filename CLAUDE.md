# Project Context: Energy Fingerprints

This project is for the Energy Data Hackdays 2026 challenge **“Energy Fingerprints: What Can You Learn from Electricity Data?”**, owned by **AEW Energie**.

Challenge page: https://www.energydatahackdays.ch/challenges/energy-fingerprints-what-can-you-learn-from-electricity-data

Challenge repository: https://gitlab.com/edhd/2026/brugg_2026/5-aew-energy-fingerprints

## Challenge

Using anonymized 15-minute smart-meter measurements, identify which energy assets and characteristic devices are present in individual households and determine when they are used. Candidate assets include:

- Electric vehicles (EVs)
- Heat pumps
- Rooftop photovoltaic (PV) systems / solar generation
- Batteries
- Other distinctive household devices

The input is an aggregated net household load signal. The challenge is to infer hidden behind-the-meter causes while preserving customer privacy.

## Data

- Four years of 15-minute electricity time-series data
- Approximately 90,000 anonymized customers
- Postal codes are included
- Around 1,000 customers have known asset labels for ground truth
- Open data, such as MeteoSwiss weather data, may be used to enrich the AEW data

## Questions to answer

1. **Asset presence:** Does a household have an EV, heat pump, rooftop PV, battery, or another distinctive device?
2. **Activity windows:** When is each asset charging, producing, heating, cycling, or discharging?
3. **Change over time:** How do fingerprints change across seasons, locations, new installations, and multiple years?

## Suggested approaches

### Device-first approach

Research published device profiles (for example, EV charging power, duration, plateau shape, and start/stop behavior), convert them into templates, and scan household time series for repeated matches.

### Label-first approach

Start with the approximately 1,000 labeled customers. Compare customers with and without a given asset, learn recurring distinguishing features, and build classifiers.

Combining signal processing, feature engineering, time-series analytics, machine learning, and modern AI techniques is encouraged.

## Expected result and evaluation

A useful result should identify the asset, locate its activity, and show the evidence. Model outputs may include asset probabilities and interpretable evidence, such as repeated EV charging plateaus, heat-pump usage correlated with falling outdoor temperature, or reduced midday consumption associated with solar generation.

Solutions are evaluated on:

- **Accuracy:** Correct asset probabilities and activity windows
- **Robustness:** Handles seasons, location, missing data, and overlapping loads
- **Explainability:** Provides inspectable evidence rather than only black-box labels
- **Scalability:** Works across multi-year data and large customer populations
- **Innovation:** Combines novel signal processing, machine learning, and AI

## Impact

Inferring asset ownership and usage patterns from smart-meter data can help utilities improve demand forecasting, infrastructure planning, tariff design, and flexibility management.

## Relevant skills

Data science and machine learning.

## Skills

Before executing a plan, always use the `grill-me` skill first. The repository-local skill is at `.agents/skills/grill-me/SKILL.md`; invoke it as `$grill-me`.

## Working guidance for the coding agent

- Treat customer data as anonymized and protect privacy in all analyses and outputs.
- Prefer interpretable, reproducible analyses with clear evidence for predictions.
- Account for seasonal effects, geographic variation, missing data, overlapping loads, and multi-year behavior.
- Document assumptions, preprocessing, features, validation strategy, and limitations.

