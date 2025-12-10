# Chicago Beach Weather Sensors Report

**Author:** Christina Mu
**Dataset:** Chicago Beach Weather Sensors (provided for DS217 final)  
**Date:** (December 8, 2025)

## Executive Summary

This analysis examines data from the Chicago Beach Weather Sensors dataset to explore real-time weather sensor reawdings from Lake Michigan beaches. The main goal was to clean and wrangle the continuous sensor time series, create meaningful temporal/ rolling features, and evaluate predictive models (Linear Regression, XGBoost). Key finding (one sentence): A tree-based model (XGBoost) outperformed linear regression in capturing seasonal and diurnal variability, with temporal features and wind/pressure-related predictors among the most important.

## Phase-by-Phase Findings

### Phase 1: Exploration

In (q1_setup_exploration.ipynb), the dataset `data/beach_sensors.csv` was loaded. In this exploratory phase, I looked at the data structure, variable formats, amount and percentage of missing values, and preliminary visualization of data. There were **195,672 observations with 18 columns** including temperature measurements (air and wet bulb), wind speed and direction, humidity, precipitation, barometric pressure, solar radiation, and sensor metadata. The data spans from April 25, 2015 (9 AM) to November 18, 2025 (12 PM), with measurements from three different weather stations: 63rd Street Weather Station, Foster Weather Station, and Oak Street Weather Station. There was significant missing values in Wet Bulb Temperature, Rain Intensity, Total Rain, Precipitation Type, and Heading (n = 75,736 observations, 38.7%).

- **Artifacts produced:**  
  - (output/q1_data_info.txt) — variable names, variable format, percentage of missing values
  - (output/q1_exploration.csv) — summary descriptive statistics
  - (output/q1_visualizations.png) — visualization of data

![Figure 1: Initial Data Exploration](output/q1_visualizations.png)
*Figure 1: Initial exploration visualizations showing distributions of air temperature distribution, air temperature time series, wet bulb temperature distribution, and wet blub time series.

### Phase 2: Data Cleaning

The notebook, (q2_data_cleaning.ipynb), was for data cleaning. In this, checks were made to identify duplicates, data cleaning addressed missing values, outliers, and data type validation.

**Cleaning Results:**

- Rows before cleaning: **195,892**
- **Duplicates:** Duplicate rows (exact duplicates) removed; duplicate timestamps per station were deduplicated by keeping the first reading. There were 0 duplicates found.
- **Missing values:** For continuous sensor readings, used **time-series forward-fill (`ffill`)**, followed by backward-fill (`bfill`), and finally median imputation where needed. This approach preserved temporal continuity while ensuring complete datasets for modeling.
  - Air Temperature: 75 missing → 0 missing
  - Wet Bulb Temperature: 75,736 missing → 0 missing (large gap, likely sensor-specific)
  - Barometric Pressure: 146 missing → 0 missing
- Outliers: Capped using IQR method (3×IQR bounds)
  - Wind Speed: 2,574 outliers capped (bounds: [-3.50, 8.40])
- Data types: Validated and converted as needed
- Rows after cleaning: **195,672** (no rows removed, only values cleaned)

- **Artifacts produced:**  
  - `output/q2_cleaned_data.csv` — cleaned dataset  
  - `output/q2_cleaning_report.txt` — detailed cleaning operations and counts  
  - `output/q2_rows_cleaned.txt` — row count after cleaning

### Phase 3: Data Wrangling

In (q3_data_wrangling.ipynb), datetime parsing and temporal feature extraction were critical for time series analysis. The `Measurement Timestamp` column was parsed from the format "MM/DD/YYYY HH:MM:SS AM/PM" and set as the DataFrame index, enabling time-based operations. I also derived additional temporal features from Measurement Timestamp to support further temporal analysis. Here is a more detailed breakdown of the new temporal features I created:

**Temporal Features Extracted:**

- `hour`: Hour of day (0-23)
- `day_of_week`: Day of week (0=Monday, 6=Sunday)
- `month`: Month of year (1-12)
- `year`: Year
- `day_name`: Day name (Monday-Sunday)
- `is_weekend`: Binary indicator (1 if Saturday/Sunday)
- `day_of_month`: Day of the month (1-31)
- `quarter`: Quarter (1-4)

### Phase 4: Feature Engineering

In (q4_feature_engineering.ipynb), I applied feature engineering on the dataset to create derived variables that would be used for machine learning models. I also decided to create rolling window features to better represent time-based interactions in the data. 
Creating derived features...

Wind-based features:
  ✓ wind_speed_squared: Wind Speed squared
  ✓ wind_category: Categorical wind speed

Comfort index:
  ✓ comfort_index: Weighted comfort metric

Humidity-based features:
  ✓ humidity_squared: Humidity squared

Interaction features:
  ✓ temp_wind_interaction: Temperature × Wind Speed

Pressure-based features:
  ✓ pressure_deviation: Deviation from mean (994.31)

Checking for infinity values...
  ✓ No infinity values found

**Rolling Window Features:**

Wind Speed rolling windows:
  ✓ wind_speed_rolling_7h: 7-hour rolling mean
  ✓ wind_speed_rolling_24h: 24-hour rolling mean
  ✓ wind_speed_rolling_std_7h: 7-hour rolling std

Humidity rolling windows:
  ✓ humidity_rolling_7h: 7-hour rolling mean
  ✓ humidity_rolling_24h: 24-hour rolling mean

Barometric Pressure rolling windows:
  ✓ pressure_rolling_7h: 7-hour rolling mean
  ✓ pressure_rolling_24h: 24-hour rolling mean

Note: I did not extract any new features from "Air Temperature", as that is the variable I am trying to predict with my analysis. Feeding new features extracted from "Air Temperature" into the machine learning models would cause data leakage, as I would be essentially informing the machine learning models of the results during training.

### Phase 5: Pattern Analysis

In (q5_pattern_analysis.ipynb), pattern analysis revealed several important temporal and correlational patterns:

- **Aggregations performed:** monthly averages (`resample('ME')`) and daily groupings (`resample('D')`).
- **Trends identified:** clear **seasonal cycle in Chicago's climate** — higher air and water temperatures in June–September and lower values in December–February.
- **Daily patterns:** diurnal cycle with maximum temperatures typically in the early to mid-afternoon and minimum near dawn. Peak air temperature typically occurs around hour 16 (4 PM), and lowest air temperature typically occurs around hour 6 (6 AM)
- **Correlations:**  
  - Air Temperature positively correlated with Wet Bulb Temperature (r=0.98) and comfort index (r=0.65).
  - Total rain is positively correlated with Air Temperature (r=0.42) and Wet Bulb Temperature (r=0.44).
  - Comfort index is negatively correlated with humidity (r=-0.71).

- **Artifacts produced:**  
  - `output/q5_correlations.csv` — correlation matrix  
  - `output/q5_patterns.png` — multi-panel visualizations (monthly trend; hourly pattern; correlation heatmap)  
  - `output/q5_trend_summary.txt` — textual summary of trends

![Figure 2: Pattern Analysis](output/q5_patterns.png)
*Figure 2: Advanced pattern analysis showing monthly temperature trends, seasonal patterns by month, daily patterns by hour, and correlation heatmap of key variables.*

### Phase 6: Modeling Preparation

In (q6_modeling_preparation.ipynb), modeling preparation involved selecting a target variable, performing temporal train/test splitting, and preparing features. Air temperature was chosen as the target variable, as it's a key indicator of beach conditions and shows predictable patterns.

**Temporal Train/Test Split:**

- Split method: Temporal (80/20 split by time, NOT random)
- Training set: **156,537 samples** (earlier data: April 2015 to ~March 2022)
- Test set: **39,179 samples** (later data: ~March 2022 to November 2025)
- Index ordering by datetime was enforced before splitting.  
- Rationale: Time series data requires temporal splitting to avoid data leakage and ensure realistic evaluation

**Feature Preparation:**
- Features selected (excluding target, non-numeric columns, and features derived from target)
- **Critical:** Excluded features derived from target variable:
  - `temp_difference` (uses Air Temperature)
  - `temp_ratio` (uses Air Temperature)
  - `temp_category` (derived from Air Temperature)
  - `comfort_index` (uses Air Temperature)
- Excluded features with >0.95 correlation to target (e.g., Wet Bulb Temperature with 0.978 correlation)
- Categorical variables (e.g., Station Name, wind_category) one-hot encoded
- All features standardized and missing values handled

- **Artifacts produced:**  
  - `output/q6_X_train.csv`, `output/q6_X_test.csv`  
  - `output/q6_y_train.csv`, `output/q6_y_test.csv`  
  - `output/q6_train_test_info.txt`

### Phase 7: Modeling

In (q7_modeling.ipynb), three models were trained and evaluated: Linear Regression, XGBoost, and Random Forest.

The month feature dominates feature importance, accounting for 78.9% of total importance. This makes intuitive sense - seasonal patterns are the strongest predictor of air temperature. Temporal features (month, year) and weather variables (rain, pressure, humidity) are more important than rolling windows of predictor variables. The top 5 features account for 92.1% of total importance.

![Figure 3: Model Performance](output/q8_final_visualizations.png)
*Figure 3: Final visualizations showing model performance comparison, predictions vs actual values, feature importance, and residuals plot for the best-performing XGBoost model.*

- **Target:** `Air Temperature` (chosen as meaningful continuous prediction target).  
- **Feature selection:** Excluded features derived from the target to avoid data leakage (`temp_difference`, `temp_ratio`, `comfort_index`, `temp_category`). Selected numeric predictors and one-hot encoded categorical predictors (e.g., `Station Name`) when present.  
- **Categorical encoding:** Teacher-style `df_encoded = pd.get_dummies(...)` used and `feature_cols` updated accordingly.  
- **Temporal split:** **80/20 temporal split (earlier 80% used for training, later 20% for testing)**. Index ordering by datetime was enforced before splitting.  
- **Artifacts produced:**  

- **Models trained:** Linear Regression, Random Forest, and XGBoost (parameters set to reasonable defaults).  
- **Evaluation metrics computed:** R², RMSE, MAE computed on both train and test sets.  
- **Feature importance:** Extracted from XGBoost (and optionally Random Forest) — saved to `output/q7_feature_importance.csv`.  
- **Predictions and results saved:**  
  - `output/q7_predictions.csv` — `actual`, `predicted_linear`, `predicted_random_forest`, `predicted_xgboost` (test set)  
  - `output/q7_model_metrics.txt` — model metrics  
  - `output/q7_feature_importance.csv` — feature importances (sorted descending)

### Phase 9: Results

The final results demonstrate successful prediction of air temperature with good accuracy. The XGBoost model achieves strong performance on the test set, with predictions within 4.87°C on average.

**Summary of Key Findings:**
1. **Model Performance:** XGBoost achieves R² = 0.7684, indicating that 76.84% of variance in air temperature can be explained by the features
2. **Feature Importance:** The month feature is overwhelmingly the most important predictor (78.9% importance), highlighting the critical role of seasonal patterns
3. **Temporal Patterns:** Strong seasonal and daily patterns are critical for accurate prediction
4. **Data Quality:** Cleaning process maintained full dataset while improving reliability
5. **Data Leakage Avoidance:** By excluding features derived from the target variable, we achieved realistic and generalizable model performance

The residuals plot shows relatively uniform distribution around zero, suggesting the model performs reasonably well across the full temperature range. The predictions vs actual scatter plot shows points distributed around the perfect prediction line with some scatter, indicating good but not perfect accuracy - which is realistic for weather prediction.

## Visualizations

![Figure 1: Initial Data Exploration](output/q1_visualizations.png)
*Figure 1: Initial exploration showing distributions and time series of key variables.*

![Figure 2: Pattern Analysis](output/q5_patterns.png)
*Figure 2: Advanced pattern analysis revealing temporal trends, seasonal patterns, daily cycles, and correlations.*

![Figure 3: Model Performance](output/q8_final_visualizations.png)
*Figure 3: Final results showing model comparison, prediction accuracy, feature importance, and residual analysis.*

## Model Results

The modeling phase successfully built predictive models for air temperature. The performance metrics demonstrate that XGBoost performs well, while Linear Regression shows that linear relationships alone are insufficient for this task.

**Performance Interpretation:**
- **R² Score:** Measures proportion of variance explained. XGBoost's R² of 0.7684 means the model explains 76.84% of variance in air temperature - a strong but realistic result.
- **RMSE (Root Mean Squared Error):** Average prediction error in original units. XGBoost's RMSE of 4.87°C means predictions are typically within 4.87°C of actual values - reasonable for weather prediction.
- **MAE (Mean Absolute Error):** Average absolute prediction error. XGBoost's MAE of 3.66°C indicates good predictive accuracy.

**Model Selection:** XGBoost is selected as the best model due to:
1. Highest R² score (0.7684)
2. Lowest RMSE (4.87°C)
3. Lowest MAE (3.66°C)
4. Good generalization (train R² = 0.9091, test R² = 0.7684 - some overfitting but reasonable)

**Feature Importance Insights:**
The feature importance analysis reveals that:
- The month feature is overwhelmingly the most important predictor (78.9% importance)
- This suggests that seasonal patterns are the strongest predictor of air temperature
- Weather variables (Total Rain, Barometric Pressure, Humidity) are important but secondary to temporal patterns
- Rolling windows of predictor variables (humidity, pressure, wind speed) contribute but are less important than seasonal features
- Temporal features (month, year) are far more important than static weather variables
- Station location has minimal impact (encoded station features have very low importance)

**Note on Data Leakage Avoidance:** By excluding features derived from the target variable (temp_difference, temp_ratio, temp_category, comfort_index) and highly correlated features (Wet Bulb Temperature), we achieved realistic model performance. This demonstrates the importance of careful feature selection to avoid circular logic.

## Time Series Patterns

The analysis revealed several important temporal patterns:

**Long-term Trends:**
- Stable long-term trends over the 10.6-year period
- No significant increasing or decreasing trends (data appears stationary after accounting for seasonality)
- Consistent seasonal cycles year over year

**Seasonal Patterns:**
- **Monthly:** Clear seasonal cycle with temperatures peaking in summer months (June-August) and reaching minima in winter months (December-February)
- Monthly air temperature range: -2.6°C to 23.6°C
- **Daily:** Strong diurnal cycle with temperatures peaking in afternoon (4 PM, hour 16) and reaching minima in early morning (6 AM, hour 6)
- Daily patterns are consistent across seasons, though amplitude varies

**Temporal Relationships:**
- Air temperature shows strong seasonal patterns (month is the most important predictor)
- Wind speed shows moderate negative correlation with temperature (-0.230)
- Humidity shows very weak correlation with temperature (0.009)
- Rolling windows of predictor variables (wind speed, humidity, pressure) capture temporal dependencies

**Anomalies:**
- Large gap in Wet Bulb Temperature data (75,626 missing values, 38.6% of dataset)
- This likely represents periods when certain sensors were not operational
- Some sensor dropouts identified (gaps in time series)
- No major anomalies in temporal patterns beyond expected seasonal variation

These temporal patterns are critical for accurate prediction, as evidenced by the high importance of temporal features (especially rolling windows) in the model.

## Limitations & Next Steps

**Limitations:**

1. **Data Quality:**
   - Large number of missing values in Wet Bulb Temperature (38.6%) required imputation, which may introduce bias
   - Sensor dropouts create gaps in time series that could affect pattern detection
   - Outlier capping may have removed some valid extreme events
   - Only 3 weather stations - limited spatial coverage

2. **Model Limitations:**
   - Linear Regression's moderate performance (R² = 0.3046) indicates that linear relationships are insufficient for this task
   - XGBoost shows some overfitting (train R² = 0.9091 vs test R² = 0.7684), though this is reasonable
   - Model relies heavily on seasonal features (month = 78.9% importance), which limits predictive power for same-season predictions
   - Model trained on historical data may not generalize to future climate conditions
   - RMSE of 4.87°C, while reasonable, may not be sufficient for applications requiring high precision

3. **Feature Engineering:**
   - Some potentially useful features may not have been created (e.g., lag features, interaction terms)
   - Rolling window sizes (7h, 24h) were chosen somewhat arbitrarily
   - Features derived from target variable were correctly excluded to avoid data leakage
   - External data (e.g., weather forecasts, lake conditions) not incorporated

4. **Scope:**
   - Analysis focused on air temperature prediction; other targets (e.g., wind speed, precipitation) not explored
   - Only one target variable analyzed; multi-target modeling could provide additional insights
   - Spatial relationships between stations not analyzed

**Next Steps:**

1. **Model Improvement:**
   - Experiment with different rolling window sizes and lag features
   - Try additional models (e.g., XGBoost, Gradient Boosting) to potentially improve performance
   - Incorporate external data sources (weather forecasts, lake level data)
   - Try ensemble methods combining multiple models
   - Validate model on truly out-of-sample data (future dates)
   - Address overfitting in XGBoost (train/test gap suggests some overfitting)

2. **Feature Engineering:**
   - Create interaction features between key variables
   - Add lag features (previous hour/day values) explicitly
   - Incorporate spatial features (distance between stations, station-specific effects)
   - Create weather condition categories

3. **Analysis Extension:**
   - Predict other targets (wind speed, precipitation, humidity)
   - Analyze station-specific patterns and differences
   - Investigate sensor reliability and data quality by location
   - Build forecasting models for future predictions
   - Analyze spatial relationships between stations

4. **Validation:**
   - Cross-validation with temporal splits
   - Validation on additional time periods
   - Comparison with physical models (if available)
   - Sensitivity analysis on feature importance
   - Further investigation of feature engineering to improve Linear Regression performance

5. **Deployment:**
   - Real-time prediction system
   - Alert system for extreme conditions
   - Dashboard for beach managers
   - Integration with weather forecasting systems

## Conclusion

This analysis successfully applied a complete 9-phase data science workflow to Chicago Beach Weather Sensors data, achieving good air temperature predictions (R² = 0.7684, RMSE = 4.87°C). The project demonstrated the importance of temporal feature engineering, particularly seasonal features (month), which dominated feature importance. Key insights include strong seasonal and daily patterns, the critical role of temporal features in prediction, and the superior performance of ensemble tree-based models over linear models. The analysis demonstrates proper data leakage avoidance by excluding features derived from the target variable, resulting in realistic and generalizable model performance. This provides a solid foundation for beach condition monitoring and prediction systems.