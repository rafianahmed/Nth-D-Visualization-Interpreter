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


def render_nd_view(df, selected_cols, domain, title_prefix="Selection", block_key="default"):
    st.markdown(f"## {title_prefix}")

    if len(selected_cols) < 2:
        st.warning("Please select at least 2 variables.")
        return

    sub = df[selected_cols].dropna()

    if sub.empty:
        st.warning("No valid rows available after removing missing values.")
        return

    st.write(f"Selected variables: {', '.join(selected_cols)}")

    # Single N-dimensional visualization
    st.subheader("N-Dimensional Relationship View")
    fig = px.parallel_coordinates(
        sub,
        dimensions=selected_cols,
        title=f"Parallel Coordinates: {' | '.join(selected_cols)}"
    )
    st.plotly_chart(fig, use_container_width=True, key=f"nd_plot_{block_key}")

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

    # Sidebar controls
    st.sidebar.header("Main View Controls")

    domain = st.sidebar.selectbox(
        "Domain Language",
        ["General", "Finance", "Supply Chain", "ML"]
    )

    n_main = st.sidebar.number_input(
        "How many variables do you want in the main view?",
        min_value=2,
        max_value=len(numeric_cols),
        value=min(4, len(numeric_cols)),
        step=1
    )

    selected_main = st.sidebar.multiselect(
        "Select variables for main view",
        options=numeric_cols,
        default=numeric_cols[:int(n_main)],
        max_selections=int(n_main)
    )

    st.sidebar.caption(f"Select exactly {int(n_main)} variables.")

    # Main analysis
    if len(selected_main) == int(n_main):
        render_nd_view(
            df=df,
            selected_cols=selected_main,
            domain=domain,
            title_prefix="Main N-Dimensional View",
            block_key="main"
        )
    else:
        st.warning(f"Please select exactly {int(n_main)} variables for the main view.")

    st.divider()

    # Comparison section
    st.header("Compare With Another Variable Set")

    n_compare = st.number_input(
        "How many variables do you want in the comparison view?",
        min_value=2,
        max_value=len(numeric_cols),
        value=min(4, len(numeric_cols)),
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

    # Current comparison preview
    if len(selected_main) == int(n_main) and len(selected_compare) == int(n_compare):
        st.divider()
        st.header("Current Comparison Preview")

        left, right = st.columns(2)

        with left:
            render_nd_view(
                df=df,
                selected_cols=selected_main,
                domain=domain,
                title_prefix="Current First View",
                block_key="current_left"
            )

        with right:
            render_nd_view(
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

    # Comparison history
    if st.session_state.comparison_history:
        st.divider()
        st.header("Comparison History")

        for idx, item in enumerate(st.session_state.comparison_history, start=1):
            st.markdown(f"## Comparison {idx}")

            left, right = st.columns(2)

            with left:
                render_nd_view(
                    df=df,
                    selected_cols=item["main_dims"],
                    domain=domain,
                    title_prefix=f"First View {idx}",
                    block_key=f"hist_left_{idx}"
                )

            with right:
                render_nd_view(
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
