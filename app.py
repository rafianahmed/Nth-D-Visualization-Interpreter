import streamlit as st
import pandas as pd
import plotly.express as px
import itertools
import numpy as np

from utils.language import enrich_language

st.set_page_config(layout="wide")
st.title("N-Dimensional Visualization Intelligence Engine")

uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

if "comparison_history" not in st.session_state:
    st.session_state.comparison_history = []


# ---------------------------
# Helper Functions
# ---------------------------
def interpret_nd(df, selected_cols):
    insights = []

    if len(selected_cols) < 2:
        return ["Please select at least 2 dimensions for interpretation."]

    desc = df[selected_cols].describe().T

    # variance-based insight
    highest_var = desc["std"].idxmax()
    lowest_var = desc["std"].idxmin()
    insights.append(f"'{highest_var}' shows the highest variability among selected dimensions.")
    insights.append(f"'{lowest_var}' is the most stable dimension among selected dimensions.")

    # correlation-based insight
    corr = df[selected_cols].corr()
    strong_pairs = []
    for i in range(len(selected_cols)):
        for j in range(i + 1, len(selected_cols)):
            c = corr.iloc[i, j]
            if abs(c) >= 0.6:
                strong_pairs.append((selected_cols[i], selected_cols[j], c))

    if strong_pairs:
        for a, b, c in strong_pairs[:5]:
            direction = "positive" if c > 0 else "negative"
            insights.append(f"'{a}' and '{b}' have a strong {direction} relationship ({c:.2f}).")
    else:
        insights.append("No very strong linear relationships were detected among the selected dimensions.")

    # range-based insight
    largest_range_col = (desc["max"] - desc["min"]).idxmax()
    insights.append(f"'{largest_range_col}' has the widest value range in the selected dimensions.")

    return insights


def confidence_score_nd(df, selected_cols):
    if len(selected_cols) < 2:
        return 0

    sub = df[selected_cols].dropna()
    if sub.empty:
        return 0

    completeness = 100 * (1 - df[selected_cols].isna().mean().mean())

    corr = sub.corr().abs()
    if len(selected_cols) > 1:
        mask = np.triu(np.ones(corr.shape), k=1).astype(bool)
        corr_vals = corr.where(mask).stack()
        avg_corr = corr_vals.mean() if not corr_vals.empty else 0
    else:
        avg_corr = 0

    variance_quality = sub.std().replace(0, np.nan).notna().mean()

    score = 0.4 * completeness + 0.35 * (avg_corr * 100) + 0.25 * (variance_quality * 100)
    return int(min(100, max(0, score)))


def compare_nd_views(df, cols_a, cols_b):
    diffs = []

    set_a = set(cols_a)
    set_b = set(cols_b)

    only_a = list(set_a - set_b)
    only_b = list(set_b - set_a)
    common = list(set_a & set_b)

    if only_a:
        diffs.append(f"Only in first selection: {', '.join(only_a)}")
    if only_b:
        diffs.append(f"Only in second selection: {', '.join(only_b)}")
    if common:
        diffs.append(f"Common dimensions: {', '.join(common)}")

    if len(common) >= 2:
        corr_a = df[common].corr().mean().mean()
        diffs.append(f"Shared-dimension average correlation structure is {corr_a:.2f}.")

    if len(cols_a) > len(cols_b):
        diffs.append("The first selection covers more dimensions and may capture broader structure.")
    elif len(cols_b) > len(cols_a):
        diffs.append("The second selection covers more dimensions and may capture broader structure.")
    else:
        diffs.append("Both selections have the same dimensional size.")

    return diffs


def plot_pairwise_views(df, selected_cols, max_pairs=6):
    pairs = list(itertools.combinations(selected_cols, 2))
    if not pairs:
        return

    st.subheader("Pairwise Views")
    for idx, (a, b) in enumerate(pairs[:max_pairs]):
        fig = px.scatter(df, x=a, y=b, title=f"{a} vs {b}")
        st.plotly_chart(fig, use_container_width=True)


def plot_3d_options(df, selected_cols):
    if len(selected_cols) < 3:
        return

    st.subheader("3D Visualization")
    c1, c2, c3 = st.columns(3)

    with c1:
        x3 = st.selectbox("3D X-axis", selected_cols, key="x3_main")
    with c2:
        y3 = st.selectbox("3D Y-axis", selected_cols, index=min(1, len(selected_cols)-1), key="y3_main")
    with c3:
        z3 = st.selectbox("3D Z-axis", selected_cols, index=min(2, len(selected_cols)-1), key="z3_main")

    fig3d = px.scatter_3d(df, x=x3, y=y3, z=z3, opacity=0.7, title=f"3D View: {x3}, {y3}, {z3}")
    st.plotly_chart(fig3d, use_container_width=True)


def render_selection_block(df, selected_cols, title_prefix="Selection"):
    st.markdown(f"## {title_prefix}")

    if len(selected_cols) < 2:
        st.warning("Please select at least 2 dimensions.")
        return

    st.write(f"Selected dimensions: {', '.join(selected_cols)}")

    # Scatter Matrix
    st.subheader("Scatter Matrix")
    fig_matrix = px.scatter_matrix(df, dimensions=selected_cols)
    fig_matrix.update_layout(height=700)
    st.plotly_chart(fig_matrix, use_container_width=True)

    # Correlation Heatmap
    st.subheader("Correlation Heatmap")
    corr = df[selected_cols].corr()
    fig_heat = px.imshow(
        corr,
        text_auto=True,
        aspect="auto",
        title="Correlation Heatmap"
    )
    st.plotly_chart(fig_heat, use_container_width=True)

    # Pairwise views
    plot_pairwise_views(df, selected_cols)

    # Optional 3D view
    plot_3d_options(df, selected_cols)

    # Interpretation
    st.subheader("Interpretation")
    raw_insights = interpret_nd(df, selected_cols)
    for item in raw_insights:
        st.markdown(f"- {item}")

    # Confidence
    st.subheader("Confidence Score")
    conf = confidence_score_nd(df, selected_cols)
    st.progress(conf / 100)
    st.caption(f"Interpretation confidence: {conf}%")


# ---------------------------
# Main App
# ---------------------------
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.dataframe(df.head())

    numeric_cols = df.select_dtypes(include="number").columns.tolist()

    if len(numeric_cols) < 2:
        st.error("Need at least 2 numeric columns.")
        st.stop()

    st.sidebar.header("Main N-Dimensional Controls")

    max_dims = len(numeric_cols)
    n_dims = st.sidebar.slider(
        "How many dimensions do you want to include?",
        min_value=2,
        max_value=max_dims,
        value=min(4, max_dims)
    )

    default_main = numeric_cols[:n_dims]
    selected_dims = st.sidebar.multiselect(
        "Select dimensions for main analysis",
        options=numeric_cols,
        default=default_main
    )

    domain = st.sidebar.selectbox(
        "Domain Language",
        ["General", "Finance", "Supply Chain", "ML"]
    )

    if len(selected_dims) >= 2:
        enriched = enrich_language(interpret_nd(df, selected_dims), domain)
        render_selection_block(df, selected_dims, "Main N-Dimensional Analysis")

        st.subheader("Domain-Enriched Interpretation")
        for item in enriched:
            st.markdown(f"- {item}")
    else:
        st.warning("Select at least 2 dimensions for the main analysis.")

    st.divider()
    st.header("Compare With Another N-Dimensional Selection")

    n_dims_2 = st.slider(
        "How many dimensions for comparison?",
        min_value=2,
        max_value=max_dims,
        value=min(4, max_dims),
        key="compare_n_dims"
    )

    default_compare = numeric_cols[:n_dims_2]
    selected_dims_2 = st.multiselect(
        "Select dimensions for comparison analysis",
        options=numeric_cols,
        default=default_compare,
        key="compare_dims"
    )

    c1, c2 = st.columns([1, 1])

    with c1:
        if st.button("Compare N-Dimensional Views"):
            if len(selected_dims) >= 2 and len(selected_dims_2) >= 2:
                diffs = compare_nd_views(df, selected_dims, selected_dims_2)
                st.session_state.comparison_history.append({
                    "main_dims": selected_dims.copy(),
                    "compare_dims": selected_dims_2.copy(),
                    "diffs": diffs
                })
            else:
                st.warning("Both selections must contain at least 2 dimensions.")

    with c2:
        if st.button("Clear Comparison History"):
            st.session_state.comparison_history = []

    if st.session_state.comparison_history:
        st.divider()
        st.header("Comparison History")

        for idx, item in enumerate(st.session_state.comparison_history, start=1):
            st.markdown(f"## Comparison {idx}")

            left, right = st.columns(2)

            with left:
                render_selection_block(df, item["main_dims"], "First Selection")

            with right:
                render_selection_block(df, item["compare_dims"], "Second Selection")

            st.subheader("Comparison Summary")
            for d in item["diffs"]:
                st.markdown(f"- {d}")

            st.markdown("---")
