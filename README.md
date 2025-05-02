Emergency Department Task Demand Forecasting
Overview
This project analyzes and forecasts task demands in an emergency department (ED) using ED_full_data.csv. It employs time-series forecasting to predict daily and weekly task demands and classification modeling to categorize task completion times, optimizing staffing and resource allocation. A future process simulation is recommended to address operational bottlenecks. Outputs include a processed dataset, visualizations, and staffing recommendations, saved in the ER_Prophet_Results directory.
Objectives

Preprocess ED data to focus on "required" tasks in pattern rooms post-July 1, 2024.
Forecast task demands using moving averages, Prophet, and Prophet + SARIMA ensemble.
Classify task completion times (short, normal, long) and predict precise times.
Recommend staffing adjustments and future simulation for bottleneck analysis.

Prerequisites

Python 3.8+
Libraries: pandas, matplotlib, seaborn, scikit-learn, prophet, numpy, xgboost, holidays, python-dateutil, statsmodels, lightgbm
Input: ED_full_data.csv
Arial font for Hebrew text support
Jupyter Notebook environment

Outputs:

Dataset: ED_pattern_rooms_only.csv
Visualizations (in ER_Prophet_Results):
Room forecasts (e.g., A101_forecast.png)
Peak demand comparison (peak_demand_forecast.png)
Daily/weekly/hourly patterns
Classification accuracy and regression errors


Text: Staffing recommendations (step_14_staffing_recommendations.txt)
Zip results: zip -r ER_Prophet_Results.zip ER_Prophet_Results



Methodology

Preprocessing:

Filter ED_full_data.csv for "required" tasks in pattern rooms (e.g., "A101") post-July 1, 2024.
Standardize timestamps, add shifts, distances, and day of week.
Save: ED_pattern_rooms_only.csv


Time-Series Forecasting:

Moving Average: 2-week window best for weekly forecasts (R²: 0.78–0.96, except "כח עזר - שונות").
Prophet: Daily room-level forecasts, capturing seasonality but weak on peaks.
Prophet + SARIMA Ensemble: Combines seasonality and peak predictions; promising but inconsistent due to limited data.
ARIMA/Exponential Smoothing: Tested but performed poorly.


Classification Modeling:

XGBoost: Predicts completion time categories (short, normal, long) for top requirements (e.g., 73% accuracy for "תפוס").
Two-Stage Model:
Stage 1: LGBMClassifier predicts time buckets (4 buckets best).
Stage 2: LGBMRegressor predicts precise times per bucket.
Moderate performance; strong for specific rooms/requirements.




Future Simulation:

Recommended process simulation to identify bottlenecks, requiring process mapping, transition probabilities, and resource allocation data.



Key Results

Forecasting: 2-week moving average excels for weekly predictions; Prophet + SARIMA shows daily potential.
Classification: XGBoost achieves 73% accuracy for "תפוס"; two-stage model effective for specific rooms.
Outputs: Processed dataset, visualizations, and staffing recommendations in ER_Prophet_Results.

Limitations

Limited data (6 months) affects model consistency.
Assumes correct CSV format.
Simplified Israeli holidays; forecasts may miss external events.
Classification limited to top requirements due to data constraints.

