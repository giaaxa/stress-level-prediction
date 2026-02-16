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
