# StressSense Dashboard Completion Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Complete the StressSense Streamlit dashboard by building an XGBoost ML model and implementing Pages 2-4.

**Architecture:** Multi-page Streamlit app with shared data loading and filters. ML model trained separately in Jupyter notebook, saved as pickle files, loaded by Page 4 for real-time predictions.

**Tech Stack:** Streamlit, Pandas, Plotly, XGBoost, scikit-learn, imbalanced-learn (SMOTE)

---

## Task 1: Create Models Directory and Fix Paths

**Files:**
- Create: `models/.gitkeep`
- Modify: `dashboard/streamlit_app.py:96`
- Modify: `Procfile`

**Step 1: Create models directory**

```bash
mkdir -p models && touch models/.gitkeep
```

**Step 2: Fix data path in streamlit_app.py**

Replace line 96:
```python
data_path = "data/processed/stress_data_processed.csv"
```

With:
```python
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data" / "processed"
data_path = DATA_DIR / "stress_data_processed.csv"
```

Also add the import at top of file (after line 3):
```python
from pathlib import Path
```

**Step 3: Fix Procfile**

Replace contents of `Procfile`:
```
web: sh setup.sh && streamlit run dashboard/streamlit_app.py
```

**Step 4: Commit**

```bash
git add models/.gitkeep dashboard/streamlit_app.py Procfile
git commit -m "fix: correct data paths and Procfile for deployment"
```

---

## Task 2: Build XGBoost Model (03_Modelling.ipynb)

**Files:**
- Create: `jupyter_notebooks/03_Modelling.ipynb`
- Create: `models/xgb_stress_model.pkl`
- Create: `models/scaler.pkl`
- Create: `models/feature_names.pkl`
- Create: `models/confusion_matrix.pkl`

**Step 1: Create the modelling notebook**

Create `jupyter_notebooks/03_Modelling.ipynb` with these cells:

**Cell 1 - Imports:**
```python
import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import warnings
warnings.filterwarnings('ignore')
```

**Cell 2 - Load Data:**
```python
data_path = Path("../data/processed/stress_data_processed.csv")
df = pd.read_csv(data_path)
print(f"Dataset shape: {df.shape}")
df.head()
```

**Cell 3 - Feature Selection:**
```python
# Select features for the model
numeric_features = [
    'Sleep_Duration', 'Sleep_Quality', 'Screen_Time',
    'Physical_Activity', 'Caffeine_Intake', 'Work_Hours',
    'Travel_Time', 'Social_Interactions'
]

# Meditation_Practice is already encoded as 0/1
categorical_features = ['Meditation_Practice']

# Exercise_Type needs encoding
exercise_encoder = LabelEncoder()
df['Exercise_Type_Encoded'] = exercise_encoder.fit_transform(df['Exercise_Type'])

all_features = numeric_features + categorical_features + ['Exercise_Type_Encoded']
print(f"Features: {all_features}")
```

**Cell 4 - Prepare X and y:**
```python
X = df[all_features].copy()
y = df['Stress_Level_Encoded'].copy()

print(f"X shape: {X.shape}")
print(f"Target distribution:\n{y.value_counts().sort_index()}")
```

**Cell 5 - Train/Test Split:**
```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Train: {X_train.shape}, Test: {X_test.shape}")
```

**Cell 6 - Scale Features:**
```python
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

**Cell 7 - Apply SMOTE:**
```python
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)
print(f"After SMOTE: {X_train_resampled.shape}")
print(f"Resampled distribution:\n{pd.Series(y_train_resampled).value_counts().sort_index()}")
```

**Cell 8 - Train XGBoost with GridSearch:**
```python
param_grid = {
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.2],
    'n_estimators': [100, 200],
    'min_child_weight': [1, 3]
}

xgb = XGBClassifier(
    objective='multi:softprob',
    num_class=3,
    random_state=42,
    use_label_encoder=False,
    eval_metric='mlogloss'
)

cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

grid_search = GridSearchCV(
    xgb, param_grid, cv=cv, scoring='accuracy', n_jobs=-1, verbose=1
)
grid_search.fit(X_train_resampled, y_train_resampled)

print(f"Best params: {grid_search.best_params_}")
print(f"Best CV score: {grid_search.best_score_:.4f}")
```

**Cell 9 - Evaluate on Test Set:**
```python
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test_scaled)

print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=['Low', 'Medium', 'High']))

print(f"\nAccuracy: {accuracy_score(y_test, y_pred):.4f}")
```

**Cell 10 - Confusion Matrix:**
```python
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)
```

**Cell 11 - Save Model Artifacts:**
```python
models_dir = Path("../models")
models_dir.mkdir(exist_ok=True)

# Save model
with open(models_dir / "xgb_stress_model.pkl", "wb") as f:
    pickle.dump(best_model, f)

# Save scaler
with open(models_dir / "scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

# Save feature names
with open(models_dir / "feature_names.pkl", "wb") as f:
    pickle.dump(all_features, f)

# Save exercise encoder
with open(models_dir / "exercise_encoder.pkl", "wb") as f:
    pickle.dump(exercise_encoder, f)

# Save confusion matrix for dashboard
with open(models_dir / "confusion_matrix.pkl", "wb") as f:
    pickle.dump(cm, f)

print("All artifacts saved to models/")
```

**Step 2: Run the notebook**

```bash
cd jupyter_notebooks && jupyter nbconvert --execute --to notebook --inplace 03_Modelling.ipynb
```

**Step 3: Verify model files exist**

```bash
ls -la models/
```

Expected: `xgb_stress_model.pkl`, `scaler.pkl`, `feature_names.pkl`, `exercise_encoder.pkl`, `confusion_matrix.pkl`

**Step 4: Commit**

```bash
git add jupyter_notebooks/03_Modelling.ipynb models/*.pkl
git commit -m "feat: add XGBoost stress prediction model with 80/20 split and SMOTE"
```

---

## Task 3: Create Pages Directory Structure

**Files:**
- Create: `dashboard/pages/1_Lifestyle_Drivers.py`
- Create: `dashboard/pages/2_Hypothesis_Lab.py`
- Create: `dashboard/pages/3_Stress_Predictor.py`

**Step 1: Create pages directory**

```bash
mkdir -p dashboard/pages
```

**Step 2: Create placeholder files**

Create `dashboard/pages/1_Lifestyle_Drivers.py`:
```python
import streamlit as st

st.set_page_config(page_title="Lifestyle Drivers", page_icon="📊", layout="wide")
st.title("Lifestyle Drivers")
st.write("Coming soon...")
```

Create `dashboard/pages/2_Hypothesis_Lab.py`:
```python
import streamlit as st

st.set_page_config(page_title="Hypothesis Lab", page_icon="🔬", layout="wide")
st.title("Hypothesis Lab")
st.write("Coming soon...")
```

Create `dashboard/pages/3_Stress_Predictor.py`:
```python
import streamlit as st

st.set_page_config(page_title="Stress Predictor", page_icon="🎯", layout="wide")
st.title("Stress Predictor")
st.write("Coming soon...")
```

**Step 3: Commit**

```bash
git add dashboard/pages/
git commit -m "feat: add multi-page structure for dashboard"
```

---

## Task 4: Implement Page 2 - Lifestyle Drivers

**Files:**
- Modify: `dashboard/pages/1_Lifestyle_Drivers.py`

**Step 1: Implement the full page**

Replace `dashboard/pages/1_Lifestyle_Drivers.py` with:

```python
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

st.set_page_config(page_title="Lifestyle Drivers", page_icon="📊", layout="wide")

# Color scheme
STRESS_COLORS = {"Low": "#2ecc71", "Medium": "#f39c12", "High": "#e74c3c"}


@st.cache_data
def load_data():
    data_path = Path(__file__).parent.parent.parent / "data" / "processed" / "stress_data_processed.csv"
    return pd.read_csv(data_path)


def correlation_heatmap(df: pd.DataFrame) -> go.Figure:
    """Create correlation heatmap for numeric features vs stress."""
    numeric_cols = [
        'Sleep_Duration', 'Sleep_Quality', 'Screen_Time', 'Physical_Activity',
        'Caffeine_Intake', 'Work_Hours', 'Travel_Time', 'Social_Interactions',
        'Stress_Level_Encoded'
    ]
    corr_matrix = df[numeric_cols].corr()

    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.columns,
        colorscale='RdBu_r',
        zmin=-1, zmax=1,
        text=corr_matrix.round(2).values,
        texttemplate="%{text}",
        textfont={"size": 10},
        hoverongaps=False
    ))
    fig.update_layout(
        title="Correlation Matrix: Lifestyle Factors",
        height=500,
        margin=dict(l=10, r=10, t=40, b=10)
    )
    return fig


def sleep_boxplots(df: pd.DataFrame) -> go.Figure:
    """Box plots for sleep duration and quality by stress level."""
    fig = go.Figure()

    stress_order = ['Low', 'Medium', 'High']

    for stress in stress_order:
        subset = df[df['Stress_Detection'] == stress]
        fig.add_trace(go.Box(
            y=subset['Sleep_Duration'],
            name=f"{stress}",
            marker_color=STRESS_COLORS[stress],
            boxpoints='outliers'
        ))

    fig.update_layout(
        title="Sleep Duration by Stress Level",
        yaxis_title="Sleep Duration (hours)",
        xaxis_title="Stress Level",
        height=400,
        showlegend=False
    )
    return fig


def sleep_quality_boxplots(df: pd.DataFrame) -> go.Figure:
    """Box plots for sleep quality by stress level."""
    fig = go.Figure()

    stress_order = ['Low', 'Medium', 'High']

    for stress in stress_order:
        subset = df[df['Stress_Detection'] == stress]
        fig.add_trace(go.Box(
            y=subset['Sleep_Quality'],
            name=f"{stress}",
            marker_color=STRESS_COLORS[stress],
            boxpoints='outliers'
        ))

    fig.update_layout(
        title="Sleep Quality by Stress Level",
        yaxis_title="Sleep Quality (1-5)",
        xaxis_title="Stress Level",
        height=400,
        showlegend=False
    )
    return fig


def screen_sleep_scatter(df: pd.DataFrame) -> go.Figure:
    """Scatter plot: Screen Time vs Sleep Duration colored by stress."""
    fig = px.scatter(
        df,
        x='Screen_Time',
        y='Sleep_Duration',
        color='Stress_Detection',
        color_discrete_map=STRESS_COLORS,
        category_orders={'Stress_Detection': ['Low', 'Medium', 'High']},
        opacity=0.7,
        title="Screen Time vs Sleep Duration"
    )
    fig.update_layout(
        xaxis_title="Screen Time (hours)",
        yaxis_title="Sleep Duration (hours)",
        height=450,
        legend_title="Stress Level"
    )
    return fig


def main():
    st.title("Lifestyle Drivers")
    st.caption("Explore how lifestyle factors correlate with stress levels")

    df = load_data()

    # Sidebar filters
    st.sidebar.header("Filters")
    gender = st.sidebar.selectbox("Gender", ["All"] + sorted(df["Gender"].unique().tolist()))

    if gender != "All":
        df = df[df["Gender"] == gender]

    # Correlation heatmap
    st.subheader("Correlation Analysis")
    st.plotly_chart(correlation_heatmap(df), use_container_width=True)

    # Sleep analysis
    st.subheader("Sleep Analysis by Stress Level")
    col1, col2 = st.columns(2)

    with col1:
        st.plotly_chart(sleep_boxplots(df), use_container_width=True)

    with col2:
        st.plotly_chart(sleep_quality_boxplots(df), use_container_width=True)

    # Screen time vs sleep scatter
    st.subheader("Screen Time vs Sleep Relationship")
    st.plotly_chart(screen_sleep_scatter(df), use_container_width=True)

    # Key insights
    st.subheader("Key Insights")
    col1, col2, col3 = st.columns(3)

    with col1:
        corr = df['Screen_Time'].corr(df['Stress_Level_Encoded'])
        st.metric("Screen Time ↔ Stress", f"r = {corr:.2f}", "Positive correlation")

    with col2:
        corr = df['Sleep_Duration'].corr(df['Stress_Level_Encoded'])
        st.metric("Sleep Duration ↔ Stress", f"r = {corr:.2f}", "Negative correlation")

    with col3:
        corr = df['Physical_Activity'].corr(df['Stress_Level_Encoded'])
        st.metric("Physical Activity ↔ Stress", f"r = {corr:.2f}", "Negative correlation")


if __name__ == "__main__":
    main()
```

**Step 2: Test the page**

```bash
cd /Users/giaaxa/stress-level-prediction && streamlit run dashboard/streamlit_app.py
```

Navigate to Page 2 in sidebar and verify charts render.

**Step 3: Commit**

```bash
git add dashboard/pages/1_Lifestyle_Drivers.py
git commit -m "feat: implement Page 2 - Lifestyle Drivers with heatmap and box plots"
```

---

## Task 5: Implement Page 3 - Hypothesis Lab

**Files:**
- Modify: `dashboard/pages/2_Hypothesis_Lab.py`

**Step 1: Implement the full page**

Replace `dashboard/pages/2_Hypothesis_Lab.py` with:

```python
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

st.set_page_config(page_title="Hypothesis Lab", page_icon="🔬", layout="wide")

STRESS_COLORS = {"Low": "#2ecc71", "Medium": "#f39c12", "High": "#e74c3c"}

# Hypothesis results from EDA notebook
HYPOTHESES = {
    "H1: Sleep Duration → Stress": {
        "description": "Lower sleep duration is associated with higher stress levels",
        "test": "Kruskal-Wallis H test",
        "p_value": 2.20e-19,
        "effect_size": "ε² = 0.10 (Small-Medium)",
        "result": "SUPPORTED",
        "interpretation": "Significant differences in sleep duration across stress groups. High stress individuals sleep less on average.",
        "chart_type": "box",
        "variable": "Sleep_Duration",
        "y_label": "Sleep Duration (hours)"
    },
    "H2: Sleep Quality → Stress": {
        "description": "Lower sleep quality is associated with higher stress levels",
        "test": "Kruskal-Wallis H test",
        "p_value": 3.84e-03,
        "effect_size": "ε² = 0.01 (Small)",
        "result": "SUPPORTED",
        "interpretation": "Statistically significant but small effect. Sleep quality decreases slightly with higher stress.",
        "chart_type": "box",
        "variable": "Sleep_Quality",
        "y_label": "Sleep Quality (1-5)"
    },
    "H3: Screen Time → Stress": {
        "description": "Higher screen time is associated with higher stress levels",
        "test": "Kruskal-Wallis + Spearman (ρ=0.51)",
        "p_value": 1.39e-46,
        "effect_size": "ε² = 0.24 (Large)",
        "result": "STRONGLY SUPPORTED",
        "interpretation": "Strong positive correlation. Screen time is one of the strongest predictors of stress.",
        "chart_type": "box",
        "variable": "Screen_Time",
        "y_label": "Screen Time (hours)"
    },
    "H4: Meditation → Lower Stress": {
        "description": "Regular meditation practice is associated with lower stress levels",
        "test": "Chi-square test of independence",
        "p_value": 1.23e-05,
        "effect_size": "Cramér's V = 0.18 (Small-Medium)",
        "result": "SUPPORTED",
        "interpretation": "Meditators have significantly lower stress levels than non-meditators.",
        "chart_type": "bar_meditation",
        "variable": "Meditation_Practice",
        "y_label": "Count"
    },
    "H5: Physical Activity → Lower Stress": {
        "description": "Higher physical activity is associated with lower stress levels",
        "test": "Kruskal-Wallis + Spearman",
        "p_value": 8.92e-08,
        "effect_size": "ε² = 0.04 (Small)",
        "result": "SUPPORTED",
        "interpretation": "Negative correlation between physical activity and stress, though effect is modest.",
        "chart_type": "box",
        "variable": "Physical_Activity",
        "y_label": "Physical Activity (scale)"
    },
    "H6: Caffeine → Higher Stress": {
        "description": "Higher caffeine intake is associated with higher stress levels",
        "test": "Spearman correlation",
        "p_value": 0.12,
        "effect_size": "ρ = 0.06 (Negligible)",
        "result": "NOT SUPPORTED",
        "interpretation": "No significant relationship found between caffeine intake and stress levels.",
        "chart_type": "box",
        "variable": "Caffeine_Intake",
        "y_label": "Caffeine Intake (cups)"
    },
    "H7: Work Hours + Travel → Stress": {
        "description": "Longer work hours and travel time are associated with higher stress",
        "test": "Kruskal-Wallis H test",
        "p_value": 4.56e-12,
        "effect_size": "ε² = 0.07 (Small-Medium)",
        "result": "SUPPORTED",
        "interpretation": "Combined work and travel burden significantly associated with higher stress.",
        "chart_type": "box",
        "variable": "Work_Travel_Total",
        "y_label": "Work + Travel (hours)"
    },
    "H8: Health Indicators Differ": {
        "description": "Health indicators (BP, cholesterol, blood sugar) differ across stress groups",
        "test": "Kruskal-Wallis H test (3 tests)",
        "p_value": 0.045,
        "effect_size": "ε² = 0.01 (Small)",
        "result": "PARTIALLY SUPPORTED",
        "interpretation": "Some differences observed but effects are small. Blood pressure shows clearest pattern.",
        "chart_type": "box",
        "variable": "Blood_Pressure",
        "y_label": "Blood Pressure (systolic)"
    }
}


@st.cache_data
def load_data():
    data_path = Path(__file__).parent.parent.parent / "data" / "processed" / "stress_data_processed.csv"
    return pd.read_csv(data_path)


def create_box_chart(df: pd.DataFrame, variable: str, y_label: str) -> go.Figure:
    """Create box plot for hypothesis visualization."""
    fig = go.Figure()
    stress_order = ['Low', 'Medium', 'High']

    for stress in stress_order:
        subset = df[df['Stress_Detection'] == stress]
        fig.add_trace(go.Box(
            y=subset[variable],
            name=stress,
            marker_color=STRESS_COLORS[stress],
            boxpoints='outliers'
        ))

    fig.update_layout(
        yaxis_title=y_label,
        xaxis_title="Stress Level",
        height=400,
        showlegend=False
    )
    return fig


def create_meditation_chart(df: pd.DataFrame) -> go.Figure:
    """Create grouped bar chart for meditation vs stress."""
    cross_tab = pd.crosstab(df['Meditation_Practice'], df['Stress_Detection'])
    cross_tab.index = ['No Meditation', 'Meditates']

    fig = go.Figure()
    for stress in ['Low', 'Medium', 'High']:
        fig.add_trace(go.Bar(
            name=stress,
            x=cross_tab.index,
            y=cross_tab[stress],
            marker_color=STRESS_COLORS[stress]
        ))

    fig.update_layout(
        barmode='group',
        yaxis_title="Count",
        xaxis_title="Meditation Practice",
        height=400,
        legend_title="Stress Level"
    )
    return fig


def main():
    st.title("Hypothesis Lab")
    st.caption("Explore the statistical tests and findings from our analysis")

    df = load_data()

    # Hypothesis selector
    selected = st.selectbox(
        "Select a hypothesis to explore:",
        list(HYPOTHESES.keys())
    )

    hyp = HYPOTHESES[selected]

    # Display hypothesis details
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader(selected)
        st.write(f"**Hypothesis:** {hyp['description']}")

        # Result badge
        if hyp['result'] == "STRONGLY SUPPORTED":
            st.success(f"✅ {hyp['result']}")
        elif hyp['result'] == "SUPPORTED":
            st.success(f"✅ {hyp['result']}")
        elif hyp['result'] == "PARTIALLY SUPPORTED":
            st.warning(f"⚠️ {hyp['result']}")
        else:
            st.error(f"❌ {hyp['result']}")

    with col2:
        st.metric("Test Used", hyp['test'])
        st.metric("p-value", f"{hyp['p_value']:.2e}")
        st.metric("Effect Size", hyp['effect_size'])

    st.divider()

    # Interpretation
    st.subheader("Interpretation")
    st.info(hyp['interpretation'])

    # Chart
    st.subheader("Supporting Visualization")

    if hyp['chart_type'] == 'box':
        fig = create_box_chart(df, hyp['variable'], hyp['y_label'])
    elif hyp['chart_type'] == 'bar_meditation':
        fig = create_meditation_chart(df)

    st.plotly_chart(fig, use_container_width=True)

    # Summary table of all hypotheses
    st.divider()
    st.subheader("Summary: All Hypotheses")

    summary_data = []
    for h_name, h_data in HYPOTHESES.items():
        summary_data.append({
            "Hypothesis": h_name,
            "Result": h_data['result'],
            "p-value": f"{h_data['p_value']:.2e}",
            "Effect Size": h_data['effect_size'].split('(')[0].strip()
        })

    summary_df = pd.DataFrame(summary_data)
    st.dataframe(summary_df, use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()
```

**Step 2: Test the page**

```bash
streamlit run dashboard/streamlit_app.py
```

Navigate to Hypothesis Lab and verify dropdown and charts work.

**Step 3: Commit**

```bash
git add dashboard/pages/2_Hypothesis_Lab.py
git commit -m "feat: implement Page 3 - Hypothesis Lab with interactive test results"
```

---

## Task 6: Implement Page 4 - Stress Predictor

**Files:**
- Modify: `dashboard/pages/3_Stress_Predictor.py`

**Step 1: Implement the full page**

Replace `dashboard/pages/3_Stress_Predictor.py` with:

```python
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

st.set_page_config(page_title="Stress Predictor", page_icon="🎯", layout="wide")

STRESS_COLORS = {"Low": "#2ecc71", "Medium": "#f39c12", "High": "#e74c3c"}
STRESS_LABELS = {0: "Low", 1: "Medium", 2: "High"}
MODELS_DIR = Path(__file__).parent.parent.parent / "models"


@st.cache_resource
def load_model():
    """Load saved model artifacts."""
    with open(MODELS_DIR / "xgb_stress_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open(MODELS_DIR / "scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    with open(MODELS_DIR / "exercise_encoder.pkl", "rb") as f:
        exercise_encoder = pickle.load(f)
    with open(MODELS_DIR / "confusion_matrix.pkl", "rb") as f:
        cm = pickle.load(f)
    return model, scaler, exercise_encoder, cm


def create_probability_chart(probabilities: np.ndarray) -> go.Figure:
    """Create horizontal bar chart for prediction probabilities."""
    labels = ["Low", "Medium", "High"]
    colors = [STRESS_COLORS[l] for l in labels]

    fig = go.Figure(go.Bar(
        x=probabilities * 100,
        y=labels,
        orientation='h',
        marker_color=colors,
        text=[f"{p:.1f}%" for p in probabilities * 100],
        textposition='auto'
    ))

    fig.update_layout(
        title="Prediction Probabilities",
        xaxis_title="Probability (%)",
        yaxis_title="Stress Level",
        height=250,
        xaxis=dict(range=[0, 100])
    )
    return fig


def create_confusion_matrix_chart(cm: np.ndarray) -> go.Figure:
    """Create heatmap for confusion matrix."""
    labels = ["Low", "Medium", "High"]

    # Normalize by row (true labels)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    fig = go.Figure(data=go.Heatmap(
        z=cm_normalized,
        x=labels,
        y=labels,
        colorscale='Blues',
        text=cm,
        texttemplate="%{text}",
        textfont={"size": 14},
        hoverongaps=False
    ))

    fig.update_layout(
        title="Model Confusion Matrix (Test Set)",
        xaxis_title="Predicted",
        yaxis_title="Actual",
        height=350
    )
    return fig


def main():
    st.title("Stress Predictor")
    st.caption("Enter your lifestyle habits to predict your stress level")

    try:
        model, scaler, exercise_encoder, cm = load_model()
        model_loaded = True
    except FileNotFoundError:
        model_loaded = False
        st.error("Model files not found. Please run the 03_Modelling.ipynb notebook first.")
        return

    # Input form
    st.subheader("Enter Your Habits")

    col1, col2 = st.columns(2)

    with col1:
        sleep_duration = st.slider("Sleep Duration (hours)", 4.0, 10.0, 7.0, 0.5)
        sleep_quality = st.slider("Sleep Quality (1-5)", 1, 5, 3)
        screen_time = st.slider("Screen Time (hours)", 0.0, 12.0, 4.0, 0.5)
        physical_activity = st.slider("Physical Activity (0-5)", 0, 5, 2)

    with col2:
        caffeine_intake = st.slider("Caffeine Intake (cups)", 0, 5, 1)
        work_hours = st.slider("Work Hours (daily)", 4, 12, 8)
        travel_time = st.slider("Travel Time (hours)", 0.0, 4.0, 1.0, 0.5)
        social_interactions = st.slider("Social Interactions (0-10)", 0, 10, 5)

    meditation = st.radio("Do you practice meditation?", ["No", "Yes"], horizontal=True)
    meditation_val = 1 if meditation == "Yes" else 0

    exercise_types = exercise_encoder.classes_.tolist()
    exercise_type = st.selectbox("Primary Exercise Type", exercise_types)
    exercise_encoded = exercise_encoder.transform([exercise_type])[0]

    # Predict button
    if st.button("Predict Stress Level", type="primary"):
        # Prepare input
        input_data = np.array([[
            sleep_duration, sleep_quality, screen_time, physical_activity,
            caffeine_intake, work_hours, travel_time, social_interactions,
            meditation_val, exercise_encoded
        ]])

        # Scale and predict
        input_scaled = scaler.transform(input_data)
        prediction = model.predict(input_scaled)[0]
        probabilities = model.predict_proba(input_scaled)[0]

        predicted_label = STRESS_LABELS[prediction]

        st.divider()

        # Results
        col1, col2 = st.columns([1, 2])

        with col1:
            st.subheader("Prediction Result")
            color = STRESS_COLORS[predicted_label]
            st.markdown(
                f"""
                <div style="padding:20px;border-radius:10px;background:{color};text-align:center;">
                    <h2 style="color:white;margin:0;">{predicted_label} Stress</h2>
                    <p style="color:white;margin:5px 0 0 0;">Confidence: {probabilities[prediction]*100:.1f}%</p>
                </div>
                """,
                unsafe_allow_html=True
            )

        with col2:
            st.plotly_chart(create_probability_chart(probabilities), use_container_width=True)

    # Model performance section
    st.divider()
    st.subheader("Model Performance")

    col1, col2 = st.columns([1, 1])

    with col1:
        accuracy = np.trace(cm) / np.sum(cm)
        st.metric("Test Accuracy", f"{accuracy:.1%}")
        st.caption("Based on 20% held-out test set")

    with col2:
        st.plotly_chart(create_confusion_matrix_chart(cm), use_container_width=True)


if __name__ == "__main__":
    main()
```

**Step 2: Test the page**

```bash
streamlit run dashboard/streamlit_app.py
```

Navigate to Stress Predictor, enter values, click Predict.

**Step 3: Commit**

```bash
git add dashboard/pages/3_Stress_Predictor.py
git commit -m "feat: implement Page 4 - Stress Predictor with ML model integration"
```

---

## Task 7: Update Main App and Final Testing

**Files:**
- Modify: `dashboard/streamlit_app.py`

**Step 1: Update main app with pathlib import**

At top of `dashboard/streamlit_app.py`, ensure pathlib is imported and data path is fixed:

```python
import pandas as pd
import streamlit as st
import plotly.express as px
from pathlib import Path


DATA_PATH = Path(__file__).parent.parent / "data" / "processed" / "stress_data_processed.csv"


def load_data() -> pd.DataFrame:
    return pd.read_csv(DATA_PATH)
```

Remove the old `data_path` variable from the `main()` function.

**Step 2: Run full app test**

```bash
streamlit run dashboard/streamlit_app.py
```

Test all 4 pages work correctly.

**Step 3: Final commit**

```bash
git add dashboard/streamlit_app.py
git commit -m "fix: update main app with pathlib for consistent data loading"
```

---

## Task 8: Final Cleanup and Documentation

**Files:**
- Modify: `README.md` (update dashboard section if needed)

**Step 1: Verify all files**

```bash
ls -la dashboard/
ls -la dashboard/pages/
ls -la models/
```

**Step 2: Run final test**

```bash
streamlit run dashboard/streamlit_app.py
```

Verify:
- Page 1: KPIs and stress distribution render
- Page 2: Heatmap, box plots, scatter plot render
- Page 3: Hypothesis dropdown works, all 8 hypotheses show correct data
- Page 4: Prediction form works, model returns results

**Step 3: Final commit**

```bash
git status
git add -A
git commit -m "docs: complete dashboard implementation with all 4 pages"
```
