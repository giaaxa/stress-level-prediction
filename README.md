#  Human Stress Detection from Sleep & Lifestyle (Streamlit)

StressSense is a Streamlit data app that explores how everyday habits (sleep, screen time, activity, work, caffeine, etc.) relate to self-reported stress level (**Low / Medium / High**).

This project is built for learning and wellbeing insight. It is **not** a medical tool and does not diagnose or treat any condition.

## CI Logo

![CI logo](https://codeinstitute.s3.amazonaws.com/fullstack/ci_logo_small.png)
---

## Dataset

**Source:** Kaggle — “Stress Level Prediction” (shijo96john)  
**Rows / Columns:** 773 rows × 22 columns  
**Target:** `Stress_Detection` (Low / Medium / High)

### Target breakdown
- Medium: 310  
- High: 301  
- Low: 162  

### What’s in the data (high level)
- **Demographics:** Age, Gender, Marital_Status, Occupation  
- **Sleep:** Sleep_Duration, Sleep_Quality, Bed_Time, Wake_Up_Time  
- **Lifestyle:** Physical_Activity, Screen_Time, Caffeine_Intake, Alcohol_Intake, Smoking_Habit, Work_Hours, Travel_Time, Social_Interactions, Meditation_Practice, Exercise_Type  
- **Health indicators:** Blood_Pressure, Cholesterol_Level, Blood_Sugar_Level  

### Notes on data cleaning
- `Bed_Time` and `Wake_Up_Time` are strings (e.g., “10:00 PM”). In ETL they’re parsed into a usable numeric form (minutes since midnight).
- No missing values or exact duplicates were found in the raw file, but ETL still enforces types and validates the dataset before analysis.

---

## Why this project exists (business rationale)

Stress is influenced by routines — sleep quality, screen habits, activity levels, work hours, commuting, and lifestyle choices. Most people know stress is a problem, but it’s not obvious which habits are most linked to it in practice.

StressSense makes those patterns visible, tests a few sensible assumptions with statistics, and demonstrates how a simple predictive feature could work in a wellbeing app.

**Target audience**
- Individuals exploring wellbeing patterns (non-clinical)
- Wellness / lifestyle app users
- Workplace wellbeing stakeholders (high-level, non-clinical)

---

## Business requirements (what the Streamlit app must answer)

1. **Stress overview**
   - How many users fall into Low / Medium / High?
   - What are the baseline averages for sleep, screen time, work hours, etc.?

2. **Key drivers**
   - Which features show the strongest relationship with stress in this dataset?

3. **Segment comparisons**
   - How does stress differ for meditation vs non-meditation, exercise types, gender, etc.?

4. **Evidence**
   - Show simple statistical tests + effect sizes where appropriate.

5. **Predictive prototype**
   - Input habits → get predicted stress category + probabilities.

6. **Clear takeaways**
   - 2–3 plain-English insights a normal human can understand.

---

## Hypotheses tested (and how I test them)

I’m not trying to “prove causation”. These are associations in this dataset.

**H1: Lower sleep duration is linked to higher stress.**  
- Test: ANOVA or Kruskal–Wallis (depending on assumptions)  
- Visual: box/violin plot of Sleep_Duration by Stress_Detection

**H2: Lower sleep quality is linked to higher stress.**  
- Test: ANOVA or Kruskal–Wallis  
- Visual: box/violin plot of Sleep_Quality by Stress_Detection

**H3: Higher screen time is linked to higher stress.**  
- Test: Spearman correlation + group comparison  
- Visual: scatter (Screen_Time vs Sleep_Duration, colour = stress) OR line-style chart of % High stress by Screen_Time bins

**H4: Meditation practice is linked to lower stress.**  
- Test: Chi-square test of independence  
- Visual: stacked/grouped bar of stress distribution by Meditation_Practice

**H5: Higher physical activity is linked to lower stress.**  
- Test: Kruskal–Wallis and/or Spearman correlation  
- Visual: box/violin of Physical_Activity by stress + optional binned trend chart

**H6: Higher caffeine intake is linked to higher stress.**  
- Test: Spearman correlation + group comparison  
- Visual: boxplot (Caffeine_Intake by stress) or binned % High stress chart

**H7: Longer work hours and longer travel time are linked to higher stress.**  
- Test: group comparison + correlation checks  
- Visual: boxplots (Work_Hours / Travel_Time by stress)

**H8: Health indicators differ across stress groups (BP, cholesterol, blood sugar).**  
- Test: ANOVA/Kruskal–Wallis across stress groups  
- Visual: boxplots for each health indicator by stress

---

## Stats & Probability (LO1 evidence)

Alongside hypothesis tests, the project includes core statistics in the notebooks:
- mean, median, variance, standard deviation
- distribution checks (normality + outliers)
- hypothesis testing logic (p-values + effect size interpretation)

A simple probability example is also included, e.g.:
- P(High Stress | Screen_Time ≥ X) vs P(High Stress | Screen_Time < X)

---

## Project approach (ETL → EDA → Stats → ML → Streamlit)

### 1) ETL (Python)
- Load raw CSV from `data/raw/`
- Enforce data types (numeric vs categorical)
- Parse `Bed_Time` and `Wake_Up_Time` → minutes since midnight
- Standardise Yes/No fields
- Reduce noisy categories where needed (e.g., Occupation Top-N + “Other”)
- Save clean dataset to `data/processed/` and export a data dictionary to `reports/`

### 2) EDA + hypothesis testing
- Summary stats + distribution plots
- Correlation scan for numeric features
- H1–H8 tests with short interpretation in plain English

### 3) Machine learning prototype (multiclass classification)
- Model: XGBoost Classifier with SMOTE for class imbalance
- Hyperparameter tuning via GridSearchCV
- Metrics: Accuracy (~67%), confusion matrix
- Model artifacts saved to `models/` directory for dashboard integration

### 4) Streamlit dashboard (the actual deliverable)
The app is split into four pages:

**Page 1 — Overview**
- KPI cards (total users, % high stress, avg sleep duration, avg screen time)
- Bar chart of stress distribution
- Filters: gender, meditation, exercise type, stress level

**Page 2 — Lifestyle Drivers**
- Correlation heatmap (numeric)
- Box/violin: sleep duration and sleep quality by stress
- Scatter: screen time vs sleep duration (colour = stress)

**Page 3 — Hypothesis Lab**
- Pick a hypothesis (H1–H8)
- Show: test used, p-value, effect size (where possible), short meaning
- Show 1 supporting chart for the selected hypothesis

**Page 4 — Stress Predictor**
- Input form (sleep, screen time, caffeine, etc.)
- Output: predicted class + probability bars
- Confusion matrix snapshot (so we're honest about performance)

> **Known Issue:** The Stress Predictor page may show an error due to XGBoost version mismatch between the training environment and the Streamlit runtime. If this occurs, retrain the model using your local environment's Python:
> ```bash
> python jupyter_notebooks/03_Modelling.ipynb  # or run the notebook cells manually
> ```

---

## Required chart types (minimum 4)
Streamlit includes at least:
- Bar chart
- Box/violin plot
- Scatter plot
- Heatmap  
(Plus optional histogram + probability bar chart)

---

## Tools used
- pandas, numpy (ETL + analysis)
- scipy.stats / statsmodels (hypothesis testing)
- scikit-learn (modelling + evaluation)
- matplotlib + plotly (visuals)
- streamlit (dashboard UI)

---

## How to run locally
1. Clone the repo
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the app:
   ```bash
   streamlit run dashboard/streamlit_app.py
   ```

---

## Deployment
Deployed using Streamlit Community Cloud.  
Live link: (add your link here)

---

## Limitations
- The dataset is cross-sectional, so it supports association, not causation.
- Bed/Wake times are self-reported averages (not tracked sleep sessions).
- Some categories (e.g. Occupation) are messy/high-cardinality and may be grouped.

---

## Ethics
- This project is not medical advice.
- Results are presented as associations in this dataset, not universal truths.
- Basic bias checks are included (e.g., model performance by gender/segments).

---

## Project structure

```
stress-level-prediction/
├── data/
│   ├── raw/                          # Original dataset
│   └── processed/                    # Cleaned & feature-engineered data
├── dashboard/
│   ├── streamlit_app.py              # Main app (Page 1 - Overview)
│   └── pages/
│       ├── 1_Lifestyle_Drivers.py    # Page 2 - Correlations & charts
│       ├── 2_Hypothesis_Lab.py       # Page 3 - Statistical tests H1-H8
│       └── 3_Stress_Predictor.py     # Page 4 - ML predictions
├── jupyter_notebooks/
│   ├── 01_ETL.ipynb                  # Data cleaning & feature engineering
│   ├── 02_EDA.ipynb                  # Exploratory analysis & hypothesis tests
│   └── 03_Modelling.ipynb            # XGBoost model training
├── models/                           # Saved model artifacts (.pkl files)
├── docs/plans/                       # Design & implementation plans
├── README.md
├── requirements.txt
├── Procfile                          # Streamlit Cloud deployment
└── setup.sh
```

---

## Credits
- Dataset: Kaggle “Stress Level Prediction” by shijo96john
- Code Institute: assessment guidance + learning materials
- ChatGPT / Codex: planning support, guidance when stuck and debugging (final edits and implementation done by the project author)
