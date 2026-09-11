# models/

| project | what |
|---|---|
| [`pv/`](pv/) | **SolarPrint** — household rooftop-PV classifier (Positive-Unlabeled, consumption-channel smart-meter features). See `pv/REPORT.md`. |
| [`heat_pump/`](heat_pump/) | Space-heating heat-pump (`has_Waermepumpe`) classifier on the GIGI property samples; selected on accuracy + confusion matrix. See `heat_pump/results.md`. |
| [`heat_pump_boiler/`](heat_pump_boiler/) | Hot-water heat-pump *boiler* detector — interpretable winter/summer plateau rule, not a trained classifier. See `heat_pump_boiler/WPB_REPORT.md`. |
