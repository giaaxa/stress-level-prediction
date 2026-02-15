import pandas as pd
import streamlit as st
import plotly.express as px
from pathlib import Path


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df


def add_sidebar_filters(df: pd.DataFrame) -> pd.DataFrame:
    st.sidebar.header("Filters")

    gender_options = ["All"] + sorted(df["Gender"].dropna().unique().tolist())
    meditation_options = ["All"] + sorted(df["Meditation_Practice"].dropna().unique().tolist())
    exercise_options = ["All"] + sorted(df["Exercise_Type"].dropna().unique().tolist())
    stress_options = ["All"] + ["Low", "Medium", "High"]

    gender = st.sidebar.selectbox("Gender", gender_options, index=0)
    meditation = st.sidebar.selectbox("Meditation Practice", meditation_options, index=0)
    exercise = st.sidebar.selectbox("Exercise Type", exercise_options, index=0)
    stress = st.sidebar.selectbox("Stress Level", stress_options, index=0)

    filtered = df.copy()
    if gender != "All":
        filtered = filtered[filtered["Gender"] == gender]
    if meditation != "All":
        filtered = filtered[filtered["Meditation_Practice"] == meditation]
    if exercise != "All":
        filtered = filtered[filtered["Exercise_Type"] == exercise]
    if stress != "All":
        filtered = filtered[filtered["Stress_Detection"] == stress]

    return filtered


def kpi_card(label: str, value: str) -> None:
    st.markdown(
        f"""
        <div style="padding:12px 16px;border:1px solid #e6e6e6;border-radius:10px;background:#fafafa;">
            <div style="font-size:12px;color:#666;margin-bottom:6px;">{label}</div>
            <div style="font-size:26px;font-weight:700;color:#111;">{value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def page_overview(df: pd.DataFrame) -> None:
    st.title("StressSense Dashboard")
    st.caption("Overview of stress distribution and baseline habits")

    col1, col2, col3, col4 = st.columns(4)

    total_users = len(df)
    pct_high = (df["Stress_Detection"] == "High").mean() * 100 if total_users else 0
    avg_sleep = df["Sleep_Duration"].mean() if total_users else 0
    avg_screen = df["Screen_Time"].mean() if total_users else 0

    with col1:
        kpi_card("Total Users", f"{total_users:,}")
    with col2:
        kpi_card("% High Stress", f"{pct_high:.1f}%")
    with col3:
        kpi_card("Avg Sleep (hrs)", f"{avg_sleep:.2f}")
    with col4:
        kpi_card("Avg Screen (hrs)", f"{avg_screen:.2f}")

    st.subheader("Stress Level Distribution")
    stress_order = ["Low", "Medium", "High"]
    stress_counts = (
        df["Stress_Detection"]
        .value_counts()
        .reindex(stress_order)
        .fillna(0)
        .reset_index()
    )
    stress_counts.columns = ["Stress Level", "Count"]

    fig = px.bar(
        stress_counts,
        x="Stress Level",
        y="Count",
        text="Count",
        color="Stress Level",
        color_discrete_map={"Low": "#2ecc71", "Medium": "#f39c12", "High": "#e74c3c"},
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(height=420, margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig, use_container_width=True)


def main() -> None:
    st.set_page_config(page_title="StressSense", page_icon="🧠", layout="wide")

    DATA_DIR = Path(__file__).parent.parent / "data" / "processed"
    data_path = DATA_DIR / "stress_data_processed.csv"
    df = load_data(data_path)

    filtered = add_sidebar_filters(df)
    page_overview(filtered)


if __name__ == "__main__":
    main()
