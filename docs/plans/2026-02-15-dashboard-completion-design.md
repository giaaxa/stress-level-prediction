# StressSense Dashboard Completion Design

## Overview

Complete the StressSense Streamlit dashboard by building the ML model and implementing Pages 2-4.

## Current State

- Page 1 (Overview) implemented: KPIs, stress distribution chart, sidebar filters
- EDA with 8 hypotheses completed in `02_EDA.ipynb`
- Processed data available at `data/processed/stress_data_processed.csv`

## Design

### Phase 1: ML Model (03_Modelling.ipynb)

**Data Preparation:**
- Features: Sleep_Duration, Sleep_Quality, Screen_Time, Physical_Activity, Caffeine_Intake, Work_Hours, Travel_Time, Meditation_Practice (encoded), Exercise_Type (encoded)
- Target: Stress_Level_Encoded (0=Low, 1=Medium, 2=High)
- Train/test split: 80/20 with stratification

**Model Pipeline:**
- StandardScaler for numeric features
- SMOTE for class imbalance
- XGBoost classifier with GridSearchCV tuning

**Outputs:**
- `models/xgb_stress_model.pkl`
- `models/scaler.pkl`
- `models/feature_names.pkl`

### Phase 2: Dashboard Pages

**Structure:**
```
dashboard/
├── streamlit_app.py          # Page 1 - Overview
└── pages/
    ├── 1_Lifestyle_Drivers.py   # Page 2
    ├── 2_Hypothesis_Lab.py      # Page 3
    └── 3_Stress_Predictor.py    # Page 4
```

**Page 2 — Lifestyle Drivers:**
- Correlation heatmap (numeric features vs stress)
- Box plots: Sleep Duration & Quality by Stress Level
- Scatter plot: Screen Time vs Sleep Duration (color = stress)

**Page 3 — Hypothesis Lab:**
- Dropdown to select H1-H8
- Display: test used, p-value, effect size, interpretation
- Supporting chart for each hypothesis

**Page 4 — Stress Predictor:**
- Input form with sliders/dropdowns for user habits
- Predict button → predicted stress level
- Probability bar chart for Low/Medium/High
- Confusion matrix heatmap

### Phase 3: Fixes

**Path Fixes:**
- Use `pathlib.Path` relative to `__file__` for cross-platform compatibility

**Procfile:**
```
web: sh setup.sh && streamlit run dashboard/streamlit_app.py
```

**Styling:**
- Consistent colors: green=Low, orange=Medium, red=High
- Sidebar filters shared via `st.session_state`
