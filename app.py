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


# =========================
# Helper Functions
# =========================
def interpret_nd(df, selected_cols):
    insights = []

    if len(selected_cols) < 2:
        return ["Please select at least 2 variables for interpretation."]

    sub = df[selected_cols].dropna()

    if sub.empty:
        return ["The selected variables do not contain enough valid data."]

    desc = sub.describe().T

    highest_var = desc["std"].idxmax()
    lowest_var = desc["std"].idxmin()
    insights.append(f"{highest_var} shows the highest variability among the selected variables.")
    insights.append(f"{lowest_var} is the most stable variable among the selected variables.")

    corr = sub.corr()
    strong_pairs = []
    for i in range(len(selected_cols)):
        for j in range(i + 1, len(selected_cols)):
            c = corr.iloc[i, j]
            if pd.notna(c) and abs(c) >= 0.6:
                strong_pairs.append((selected_cols[i], selected_cols[j], c))

    if strong_pairs:
        for a, b, c in strong_pairs[:5]:
            direction = "positive" if c > 0 else "negative"
            insights.append(f"{a} and {b} show a strong {direction} relationship ({c:.2f}).")
    else:
        insights.append("No very strong linear relationships were found among the selected variables.")

    ranges = desc["max"] - desc["min"]
    largest_range_col = ranges.idxmax()
    insights.append(f"{largest_range_col} has the widest value range.")

    return insights


def confidence_score_nd(df, selected_cols):
    if len(selected_cols) < 2:
        return 0

    sub = df[selected_cols].dropna()
    if sub.empty:
        return 0

    completeness = 100 * (1 - df[selected_cols].isna().mean().mean())

    corr = sub.corr().abs()
    mask = np.triu(np.ones(corr.shape), k=1).astype(bool)
    corr_vals = corr.where(mask).stack()
    avg_corr = corr_vals.mean() if not corr_vals.empty else 0

    variance_quality = sub.std().replace(0, np.nan).notna().mean()

    score = 0.4 * completeness + 0.35 * (avg_corr * 100) + 0.25 * (variance_quality * 100)
    return int(min(100, max(0, score)))


def compare_nd_views(df, cols_a, cols_b):
    diffs = []

    set_a = set(cols_a)
    set_b = set(cols_b)

    only_a = sorted(list(set_a - set_b))
    only_b = sorted(list(set_b - set_a))
    common = sorted(list(set_a & set_b))

    if only_a:
        diffs.append(f"Only in first selection: {', '.join(only_a)}")
    if only_b:
        diffs.append(f"Only in second selection: {', '.join(only_b)}")
    if common:
        diffs.append(f"Common variables: {', '.join(common)}")

    if len(cols_a) > len(cols_b):
        diffs.append("The first view has more variables.")
    elif len(cols_b) > len(cols_a):
        diffs.append("The second view has more variables.")
    else:
        diffs.append("Both views use the same number of variables.")

    shared = [c for c in cols_a if c in cols_b]
    if len(shared) >= 2:
        shared_corr = df[shared].dropna().corr().abs()
        mask = np.triu(np.ones(shared_corr.shape), k=1).astype(bool)
        vals = shared_corr.where(mask).stack()
        if not vals.empty:
            diffs.append(f"The shared variables have an average absolute correlation of {vals.mean():.2f}.")

    return diffs


def plot_pairwise_views(df, selected_cols, block_key):
    pairs = list(itertools.combinations(selected_cols, 2))

    if not pairs:
        return

    st.subheader("Pairwise Scatter Plots")

    max_pairs = min(len(pairs), 6)
    for idx in range(max_pairs):
        a, b = pairs[idx]
        fig = px.scatter(df, x=a, y=b, title=f"{a} vs {b}")
        st.plotly_chart(fig, use_container_width=True, key=f"pair_{block_key}_{idx}")


def plot_optional_3d(df, selected_cols, block_key):
    if len(selected_cols) < 3:
        return

    st.subheader("3D View")
    c1, c2, c3 = st.columns(3)

    with c1:
        x3 = st.selectbox("3D X-axis", selected_cols, key=f"x3_{block_key}")
    with c2:
        y3 = st.selectbox(
            "3D Y-axis",
            selected_cols,
            index=min(1, len(selected_cols) - 1),
            key=f"y3_{block_key}"
        )
    with c3:
        z3 = st.selectbox(
            "3D Z-axis",
            selected_cols,
            index=min(2, len(selected_cols) - 1),
            key=f"z3_{block_key}"
        )

    fig3d = px.scatter_3d(
        df,
        x=x3,
        y=y3,
        z=z3,
        opacity=0.7,
        title=f"{x3} vs {y3} vs {z3}"
    )
    st.plotly_chart(fig3d, use_container_width=True, key=f"plot3d_{block_key}")


def render_selection_block(df, selected_cols, title_prefix="Selection", block_key="default"):
    st.markdown(f"## {title_prefix}")

    if len(selected_cols) < 2:
        st.warning("Please select at least 2 variables.")
        return

    st.write(f"Selected variables: {', '.join(selected_cols)}")

    st.subheader("Scatter Matrix")
    fig_matrix = px.scatter_matrix(df, dimensions=selected_cols)
    fig_matrix.update_layout(height=700)
    st.plotly_chart(fig_matrix, use_container_width=True, key=f"matrix_{block_key}")

    st.subheader("Correlation Heatmap")
    corr = df[selected_cols].corr()
    fig_heat = px.imshow(
        corr,
        text_auto=True,
        aspect="auto",
        title="Correlation Heatmap"
    )
    st.plotly_chart(fig_heat, use_container_width=True, key=f"heat_{block_key}")

    plot_pairwise_views(df, selected_cols, block_key)
    plot_optional_3d(df, selected_cols, block_key)

    st.subheader("Interpretation")
    raw_insights = interpret_nd(df, selected_cols)
    for item in raw_insights:
        st.markdown(f"- {item}")

    st.subheader("Confidence Score")
    conf = confidence_score_nd(df, selected_cols)
    st.progress(conf / 100)
    st.caption(f"Interpretation confidence: {conf}%")


# =========================
# Main App
# =========================
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.dataframe(df.head())

    numeric_cols = df.select_dtypes(include="number").columns.tolist()

    if len(numeric_cols) < 2:
        st.error("Need at least 2 numeric columns.")
        st.stop()

    st.sidebar.header("Main View Controls")

    n_main = st.sidebar.number_input(
        "How many variables do you want in the main view?",
        min_value=2,
        max_value=len(numeric_cols),
        value=min(3, len(numeric_cols)),
        step=1
    )

    default_main = numeric_cols[:int(n_main)]

    selected_dims = st.sidebar.multiselect(
        "Select variables for main view",
        options=numeric_cols,
        default=default_main,
        max_selections=int(n_main)
    )

    st.sidebar.caption(f"Please select exactly {int(n_main)} variables.")

    domain = st.sidebar.selectbox(
        "Domain Language",
        ["General", "Finance", "Supply Chain", "ML"]
    )

    if len(selected_dims) == int(n_main):
        render_selection_block(
            df,
            selected_dims,
            "Main Analysis",
            block_key="main"
        )

        st.subheader("Domain-Enriched Interpretation")
        enriched = enrich_language(interpret_nd(df, selected_dims), domain)
        for item in enriched:
            st.markdown(f"- {item}")
    else:
        st.warning(f"Please select exactly {int(n_main)} variables for the main view.")

    st.divider()
    st.header("Compare With Another Variable Set")

    n_compare = st.number_input(
        "How many variables do you want in the comparison view?",
        min_value=2,
        max_value=len(numeric_cols),
        value=min(3, len(numeric_cols)),
        step=1,
        key="n_compare"
    )

    default_compare = numeric_cols[:int(n_compare)]

    selected_dims_2 = st.multiselect(
        "Select variables for comparison view",
        options=numeric_cols,
        default=default_compare,
        max_selections=int(n_compare),
        key="compare_dims"
    )

    st.caption(f"Please select exactly {int(n_compare)} variables for the comparison view.")

    c1, c2 = st.columns(2)

    with c1:
        if st.button("Compare Views"):
            if len(selected_dims) == int(n_main) and len(selected_dims_2) == int(n_compare):
                diffs = compare_nd_views(df, selected_dims, selected_dims_2)
                st.session_state.comparison_history.append({
                    "main_dims": selected_dims.copy(),
                    "compare_dims": selected_dims_2.copy(),
                    "diffs": diffs
                })
            else:
                st.warning("Please select the required number of variables in both views.")

    with c2:
        if st.button("Clear History"):
            st.session_state.comparison_history = []

    if st.session_state.comparison_history:
        st.divider()
        st.header("Comparison History")

        for idx, item in enumerate(st.session_state.comparison_history, start=1):
            st.markdown(f"## Comparison {idx}")

            left, right = st.columns(2)

            with left:
                render_selection_block(
                    df,
                    item["main_dims"],
                    title_prefix=f"First View {idx}",
                    block_key=f"hist_left_{idx}"
                )

            with right:
                render_selection_block(
                    df,
                    item["compare_dims"],
                    title_prefix=f"Second View {idx}",
                    block_key=f"hist_right_{idx}"
                )

            st.subheader("Comparison Summary")
            for d in item["diffs"]:
                st.markdown(f"- {d}")

            st.markdown("---")
else:
    st.info("Upload a CSV file to begin analysis.")
