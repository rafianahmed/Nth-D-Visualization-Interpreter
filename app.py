import itertools

import pandas as pd
import plotly.express as px
import streamlit as st

from utils.comparator import compare_views
from utils.confidence import confidence_score_nd
from utils.interpreter import interpret_nd
from utils.language import enrich_language

st.set_page_config(layout="wide", page_title="N-Dimensional Visualization Intelligence Engine")
st.title("N-Dimensional Visualization Intelligence Engine")

uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

if "comparison_history" not in st.session_state:
    st.session_state.comparison_history = []


def plot_pairwise_views(df, selected_cols, block_key, max_pairs=6):
    pairs = list(itertools.combinations(selected_cols, 2))

    if not pairs:
        return

    st.subheader("Pairwise Scatter Plots")

    for idx, (a, b) in enumerate(pairs[:max_pairs]):
        fig = px.scatter(
            df,
            x=a,
            y=b,
            title=f"{a} vs {b}",
            opacity=0.75
        )
        st.plotly_chart(fig, use_container_width=True, key=f"pair_{block_key}_{idx}")


def plot_optional_3d(df, selected_cols, block_key):
    if len(selected_cols) < 3:
        return

    st.subheader("3D View")

    c1, c2, c3 = st.columns(3)

    with c1:
        x3 = st.selectbox(
            "3D X-axis",
            selected_cols,
            index=0,
            key=f"x3_{block_key}"
        )

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


def render_selection_block(df, selected_cols, domain, title_prefix="Selection", block_key="default"):
    st.markdown(f"## {title_prefix}")

    if len(selected_cols) < 2:
        st.warning("Please select at least 2 variables.")
        return

    st.write(f"Selected variables: {', '.join(selected_cols)}")

    # Scatter Matrix
    st.subheader("Scatter Matrix")
    fig_matrix = px.scatter_matrix(df, dimensions=selected_cols)
    fig_matrix.update_layout(height=700)
    st.plotly_chart(fig_matrix, use_container_width=True, key=f"matrix_{block_key}")

    # Correlation Heatmap
    st.subheader("Correlation Heatmap")
    corr = df[selected_cols].corr(numeric_only=True)
    fig_heat = px.imshow(
        corr,
        text_auto=True,
        aspect="auto",
        title="Correlation Heatmap"
    )
    st.plotly_chart(fig_heat, use_container_width=True, key=f"heat_{block_key}")

    # Pairwise views
    plot_pairwise_views(df, selected_cols, block_key)

    # Optional 3D
    plot_optional_3d(df, selected_cols, block_key)

    # Interpretation
    st.subheader("Interpretation")
    raw_insights = interpret_nd(df, selected_cols)
    enriched_insights = enrich_language(raw_insights, domain)

    for insight in enriched_insights:
        st.markdown(f"- {insight}")

    # Confidence
    st.subheader("Confidence Score")
    conf = confidence_score_nd(df, selected_cols)
    st.progress(conf / 100)
    st.caption(f"Interpretation confidence: {conf}%")


if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"Failed to read CSV: {e}")
        st.stop()

    st.subheader("Preview")
    st.dataframe(df.head(), use_container_width=True)

    numeric_cols = df.select_dtypes(include="number").columns.tolist()

    if len(numeric_cols) < 2:
        st.error("Need at least 2 numeric columns.")
        st.stop()

    # -------------------------
    # Sidebar controls
    # -------------------------
    st.sidebar.header("Main View Controls")

    domain = st.sidebar.selectbox(
        "Domain Language",
        ["General", "Finance", "Supply Chain", "ML"]
    )

    n_main = st.sidebar.number_input(
        "How many variables do you want in the main view?",
        min_value=2,
        max_value=len(numeric_cols),
        value=min(3, len(numeric_cols)),
        step=1
    )

    selected_main = st.sidebar.multiselect(
        "Select variables for main view",
        options=numeric_cols,
        default=numeric_cols[:int(n_main)],
        max_selections=int(n_main)
    )

    st.sidebar.caption(f"Select exactly {int(n_main)} variables.")

    # -------------------------
    # Main analysis
    # -------------------------
    if len(selected_main) == int(n_main):
        render_selection_block(
            df=df,
            selected_cols=selected_main,
            domain=domain,
            title_prefix="Main Analysis",
            block_key="main"
        )
    else:
        st.warning(f"Please select exactly {int(n_main)} variables for the main view.")

    st.divider()

    # -------------------------
    # Comparison section
    # -------------------------
    st.header("Compare With Another Variable Set")

    n_compare = st.number_input(
        "How many variables do you want in the comparison view?",
        min_value=2,
        max_value=len(numeric_cols),
        value=min(3, len(numeric_cols)),
        step=1,
        key="n_compare"
    )

    selected_compare = st.multiselect(
        "Select variables for comparison view",
        options=numeric_cols,
        default=numeric_cols[:int(n_compare)],
        max_selections=int(n_compare),
        key="compare_dims"
    )

    st.caption(f"Select exactly {int(n_compare)} variables for the comparison view.")

    c1, c2 = st.columns(2)

    with c1:
        if st.button("Compare Views"):
            if len(selected_main) == int(n_main) and len(selected_compare) == int(n_compare):
                diffs = compare_views(df, selected_main, selected_compare)
                st.session_state.comparison_history.append(
                    {
                        "main_dims": selected_main.copy(),
                        "compare_dims": selected_compare.copy(),
                        "diffs": diffs,
                    }
                )
            else:
                st.warning("Please select the required number of variables in both views.")

    with c2:
        if st.button("Clear History"):
            st.session_state.comparison_history = []

    # -------------------------
    # Current comparison preview
    # -------------------------
    if len(selected_main) == int(n_main) and len(selected_compare) == int(n_compare):
        st.divider()
        st.header("Current Comparison Preview")

        left, right = st.columns(2)

        with left:
            render_selection_block(
                df=df,
                selected_cols=selected_main,
                domain=domain,
                title_prefix="Current First View",
                block_key="current_left"
            )

        with right:
            render_selection_block(
                df=df,
                selected_cols=selected_compare,
                domain=domain,
                title_prefix="Current Second View",
                block_key="current_right"
            )

        st.subheader("Current Comparison Summary")
        current_diffs = compare_views(df, selected_main, selected_compare)
        for d in current_diffs:
            st.markdown(f"- {d}")

    # -------------------------
    # Comparison history
    # -------------------------
    if st.session_state.comparison_history:
        st.divider()
        st.header("Comparison History")

        for idx, item in enumerate(st.session_state.comparison_history, start=1):
            st.markdown(f"## Comparison {idx}")

            left, right = st.columns(2)

            with left:
                render_selection_block(
                    df=df,
                    selected_cols=item["main_dims"],
                    domain=domain,
                    title_prefix=f"First View {idx}",
                    block_key=f"hist_left_{idx}"
                )

            with right:
                render_selection_block(
                    df=df,
                    selected_cols=item["compare_dims"],
                    domain=domain,
                    title_prefix=f"Second View {idx}",
                    block_key=f"hist_right_{idx}"
                )

            st.subheader("Comparison Summary")
            for d in item["diffs"]:
                st.markdown(f"- {d}")

            st.markdown("---")

else:
    st.info("Upload a CSV file to begin analysis.")
