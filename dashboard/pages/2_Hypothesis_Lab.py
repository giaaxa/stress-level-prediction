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
