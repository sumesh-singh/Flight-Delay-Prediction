# Project Audit Report: Against Abstract Claims

## Executive Summary
The project successfully implements the core **Flight Delay Prediction System** described in the abstract, with strong adherence to **Data Integration** (BTS, NOAA, OurAirports, OpenSky), **Feature Engineering** (Temporal, Network, Human Factors), and **Reproducibility** (Docker, Scripts). However, two significant components mentioned in the abstract — **Regression Task (Delay Magnitude)** and **Active Outlier Detection** — are implemented as modules but not integrated into the main training pipeline. The reported performance (F1=0.79) in the abstract likely reflects results prior to the leakage remediation (current realistic F1 ≈ 0.54).

## Detailed Audit

| Abstract Claim | Status | Implementation Details |
| :--- | :--- | :--- |
| **Data Sources** | | |
| "BTS On-Time Performance dataset" | ✅ **Implemented** | `MultiYearDataLoader` handles 2023-2025 data. |
| "Airport metadata from OurAirports" | ✅ **Implemented** | `AirportMapper` uses `data/metadata/airports.csv` to map IATA→ICAO. |
| "Weather data from NOAA" | ✅ **Implemented** | `NOAAClient` fetches daily weather for Origin/Dest airports. |
| **Preprocessing** | | |
| "Outlier detection using Isolation Forest, Local Outlier Factor" | ⚠️ **Partial** | Implemented in `DataCleanser` (`clean_data` method) but **NOT called** in `train_pipeline.py`. |
| "Standardized preprocessing" | ✅ **Implemented** | `train_pipeline.py` ensures consistent scaling and encoding. |
| **Feature Engineering** | | |
| "Temporal attributes (cyclical encoding)" | ✅ **Implemented** | `create_temporal_features` (sin/cos for hour, day, month). |
| "Airline characteristics (historical delay)" | ✅ **Implemented** | `create_carrier_features` computes rolling stats. |
| "Network propagation effects" | ✅ **Implemented** | `create_network_features` tracks tail-number turnarounds. |
| "Human factors (crew fatigue, ATC workload)" | ✅ **Implemented** | `create_human_factors_features` (newly added). |
| **Prediction Task** | | |
| "Binary classification (15-min threshold)" | ✅ **Implemented** | `IS_DELAYED` target variable used in all models. |
| "Multiclass regression (delay magnitude)" | ❌ **Missing** | No regression models (e.g., `RandomForestRegressor`) in `train_pipeline.py`. |
| "Random Forest ... Logistic Regression" | ✅ **Implemented** | Both models trained + SGD Classifier. |
| **Performance** | | |
| "Random Forest achieves 83.2% accuracy and 0.79 F1-score" | ⚠️ **Discrepancy** | Current best: **81.7% Accuracy, 0.54 F1**. <br>The abstract's 0.79 F1 likely relied on leakage features (e.g. `ArrivalDelayGroups`) which we removed to ensure validity. |
| "RMSE of 17.3 minutes" | ❌ **Missing** | Regression metrics not computed in main pipeline. |
| **Reproducibility** | | |
| "Open-source, comprehensive documentation" | ✅ **Implemented** | Root `README.md`, `Dockerfile`, `requirements.txt` fully set up. |
| "Modular architecture" | ✅ **Implemented** | `src/` organized by data, features, models. |

## Recommendations

1.  **Integrate Outlier Detection**: Enable `DataCleanser` in `train_pipeline.py` to remove anomalies (might improve F1 slightly).
2.  **Add Regression Task**: Create a `train_regression_pipeline.py` to predict `ARR_DELAY` (minutes) using `RandomForestRegressor`, effectively fulfilling the "delay magnitude" claim.
3.  **Update Abstract**: Adjust the reported F1 score to the realistic **0.54** (unleaked) or highlight that 0.79 was achieved with "extended features" (leakage). Honesty in research is ensuring reproducibility.

## Conclusion
The project is **~85% aligned** with the abstract. The missing 15% (Regression, Outlier Detection integration) does not block the primary goal of binary delay prediction, which is fully functional and deployment-ready.
