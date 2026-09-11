# Literature Feature Extraction Matrix for Behind-the-Meter Asset Detection

This document summarizes key research literature on behind-the-meter (BTM) asset identification (Photovoltaics, Electric Vehicles, Heat Pumps, and Battery Energy Storage Systems) using low-frequency smart meter time-series data (15-min to 60-min resolution) and metadata.

---

## Literature Feature Extraction Matrix

| Paper & Reference | Target Asset(s) | Explicit Feature Mapping by Target Variable | Data Resolution & Method |
|---|---|---|---|
| **Vrba et al. (2024)** | PV, EV, Heat Pump | • **EV:** Min-to-median daily energy spread (`min-median`), standard deviation of minimum power drawn (`std-maximum`), non-business hours net load variance.<br>• **PV:** Negative load magnitude, daily min-to-max load ratio.<br>• **PV / EV:** Autocorrelation at lags 1 and 10 ($ACF_1, ACF_{10}$).<br>• **Heat Pump:** Seasonal baseline variance. | **15-min** smart meter data<br>• LightGBM & Random Forest |
| **Hoffmann & Nytun (SINTEF, 2023)** | EV | • **EV:** High-power step magnitude during off-peak hours ($\ge 3.7\text{ kW}$), nocturnal rolling mean & standard deviation (22:00–06:00), peak-to-median power ratio, weekend vs. weekday night load variance spread. | **60-min** smart meter data<br>• Random Forest & LightGBM |
| **Bojer et al. (2023)** | Heat Pump | • **Heat Pump:** Daily power vs. ambient temperature slope ($\Delta P / \Delta T$), Heating Degree Day (HDD) correlation ($R^2$), winter-to-summer energy ratio ($E_{\text{winter}}/E_{\text{summer}}$), winter quantile distributions ($Q_{10}, Q_{50}, Q_{90}$). | **15-min / 60-min** data + weather<br>• Random Forest & 1D-CNN |
| **Müller et al. (2022)** | PV, EV | • **PV:** Negative power duration ($P < 0\text{ kW}$), time-of-day power probability distributions.<br>• **EV:** Step duration above threshold ($\ge 3.7\text{ kW}$ or $5\text{ kW}$), nighttime peak load amplitude, Kernel Density Estimation (KDE) feature vectors. | **30-min & 60-min** smart meter data<br>• Combined CNN + MLP |
| **Arefin et al. (2023)** | EV | • **EV:** Sliding temporal window power vectors, local density deviation of load spikes, off-peak (19:00–06:00) power standard deviation, sustained high-power duration ($> 3.0\text{ kW}$). | **15-min & 30-min** smart meter data<br>• CNN + GRU (ODEV) |
| **Althaher et al. (2023)** | PV, Battery (BESS) | • **PV:** Net export depth ($P_{\min} < 0\text{ kW}$), solar irradiance-to-net load cross-correlation.<br>• **BESS:** Active-to-reactive power ratio ($P/Q$), zero-load plateau duration ($\pm 0.1\text{ kW}$ during daylight hours), intraday peak-shaving variance reduction. | **15-min** smart meter + GHI weather<br>• XGBoost & Decision Trees |
| **Zhao et al. (2022)** | Heat Pump, PV | • **Heat Pump:** Temperature-dependent load sensitivity coefficient ($\Delta P / \Delta T$ below $15^\circ\text{C}$), seasonal baseline shift index.<br>• **PV:** Midday consumption skewness (11:00–15:00 vs. 01:00–05:00), 24-hour autocorrelation periodicity ($ACF_{24}$). | **15-min** smart meter + weather data<br>• LightGBM & Feature Selection |
| **Beckel et al. (ETH Zurich, 2014)** | Heat Pump, EV | • **Heat Pump:** Consumption quantiles ($Q_{10}, Q_{50}, Q_{90}$), baseload ratio ($Q_{10} / Q_{50}$), day-to-night consumption ratio.<br>• **EV:** Skewness & kurtosis of daily load profiles, weekday-to-weekend ratio, 24-hour autocorrelation ($ACF_{24}$). | **30-min** smart meter data<br>• SVM & Random Forest |
| **Wang et al. (ACM KDD, 2013)** | Heat Pump | • **Heat Pump:** Daily power vs. ambient temperature slope ($\Delta P / \Delta T$), Heating Degree Day correlation ($R^2$), winter wavelet transform energy coefficients, seasonal baseline consumption shift. | **Daily / 1-hour** data + weather<br>• Biased SVM (PU Learning) |
| **Le Ray et al. (IEEE, 2019)** | Heat Pump | • **Heat Pump:** Boxcar approximation pulse vectors, duty-cycle step magnitudes, operating duration per temperature band, active power pulse probability distributions. | **5-min & 15-min** smart meter data<br>• Bayesian Orthogonal Matching Pursuit |
| **Kabir et al. (IEEE, 2021)** | PV, Battery (BESS) | • **PV:** Solar irradiance-to-net load Pearson correlation, afternoon power ramp rate ($\Delta P / \Delta t$), net export depth ($P_{\min} < 0\text{ kW}$).<br>• **BESS:** Zero-load plateau duration ($\pm 0.1\text{ kW}$ during daylight hours). | **15-min** smart meter + GHI weather<br>• XGBoost & Decision Trees |
| **Martin et al. (2024)** | EV | • **EV:** Sliding-window step magnitude ($\ge 3.7\text{ kW}$ or $7.4\text{ kW}$), off-peak (19:00–06:00) power standard deviation, continuous high-power state duration ($\ge 2\text{ hours}$), min-to-peak power ratio. | **15-min** smart meter data<br>• Sliding-Window Random Forest |

---

## Summary of Consolidated Feature Sets by Target Asset

### 1. Photovoltaic (PV) Identification
* **Net Export Depth & Frequency:** Share of time intervals with negative active power ($P < 0\text{ kW}$) and magnitude of $P_{\min}$.
* **Solar Irradiance Correlation:** Pearson/Spearman correlation between net hourly load and Global Horizontal Irradiance (GHI from MeteoSwiss / Open-Meteo) during peak solar window (10:00–16:00).
* **Diurnal Skewness:** Midday-to-night power ratio $\frac{P_{11:00-15:00}}{P_{01:00-05:00}}$.
* **Periodic Autocorrelation:** 24-hour lag autocorrelation ($ACF_{24}$) reflecting clear diurnal solar cycles.

### 2. Electric Vehicle (EV) Identification
* **High-Power Step Jumps:** Frequency and duration of power steps $\ge 3.7\text{ kW}$ or $7.4\text{ kW}$ sustained over $\ge 2\text{ hours}$.
* **Off-Peak Nighttime Volatility:** Standard deviation and $Q_{95}-Q_{99}$ percentiles calculated exclusively during nighttime hours (19:00–06:00).
* **Min-to-Median Spread:** Variance between $P_{\min}$ and $P_{\median}$, separating high-power intermittent EV sessions from baseline domestic load.

### 3. Heat Pump (HP) Identification
* **Thermal Sensitivity Gradient ($\Delta P / \Delta T$):** Linear regression slope of daily energy consumption against outdoor temperature below $15^\circ\text{C}$.
* **HDD Goodness of Fit ($R^2$):** Coefficient of determination against Heating Degree Days.
* **Seasonal Energy Ratio ($E_{\text{winter}} / E_{\text{summer}}$):** Total energy ratio between winter months (Nov–Feb) and summer months (Jun–Aug).
* **Metadata Coupling (GWR / RegBL):** Direct feature incorporation of Swiss Federal Register building heating energy code (`GHEIZ`).

### 4. Battery Storage System (BSS) Identification
* **Zero-Load Plateau Duration:** Cumulative daylight hours where net active power stays near zero ($\pm 0.1\text{ kW}$) despite high solar GHI.
* **Peak Tariff Variance Suppression:** Reduced load variance during peak grid pricing hours due to automated discharge.
* **Abrupt State Transitions:** Sharp step shifts from full charge rate to zero power when battery capacity is reached.

behind_the_meter_asset_detection_literature.md
behind_the_meter_asset_detection_literature.md wird angezeigt.