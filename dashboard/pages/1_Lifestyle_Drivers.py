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
