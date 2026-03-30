import streamlit as st
import pandas as pd
import plotly.express as px

from utils.interpreter import interpret_3d
from utils.comparator import compare_views
from utils.language import enrich_language
from utils.confidence import confidence_score

st.set_page_config(layout="wide")
st.title("3D Visualization Intelligence Engine")

uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

# Store comparison history
if "comparison_history" not in st.session_state:
    st.session_state.comparison_history = []

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.dataframe(df.head())

    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    if len(numeric_cols) < 3:
        st.error("Need at least 3 numeric columns.")
        st.stop()

    st.sidebar.header("3D View Controls")
    x = st.sidebar.selectbox("X-axis", numeric_cols)
    y = st.sidebar.selectbox("Y-axis", numeric_cols, index=1)
    z = st.sidebar.selectbox("Z-axis", numeric_cols, index=2)

    domain = st.sidebar.selectbox(
        "Domain Language",
        ["General", "Finance", "Supply Chain", "ML"]
    )

    # Main selected 3D view
    st.subheader("📌 Current Main 3D View")
    fig = px.scatter_3d(df, x=x, y=y, z=z, opacity=0.7, title=f"Main View: {x} vs {y} vs {z}")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("🧠 Interpretation")
    raw_insights = interpret_3d(df, x, y, z)
    enriched = enrich_language(raw_insights, domain)

    for i in enriched:
        st.markdown(f"- {i}")

    st.subheader("📊 Confidence Scores")
    conf = confidence_score(df, x, y, z)
    st.progress(conf / 100)
    st.caption(f"Interpretation confidence: {conf}%")

    st.divider()
    st.subheader("🔄 Compare With Another View")

    x2 = st.selectbox("X-axis (comparison)", numeric_cols, key="x2")
    y2 = st.selectbox("Y-axis (comparison)", numeric_cols, index=1, key="y2")
    z2 = st.selectbox("Z-axis (comparison)", numeric_cols, index=2, key="z2")

    col_btn1, col_btn2 = st.columns([1, 1])

    with col_btn1:
        if st.button("Compare Views"):
            diffs = compare_views(df, (x, y, z), (x2, y2, z2))

            comparison_entry = {
                "main_axes": (x, y, z),
                "comp_axes": (x2, y2, z2),
                "diffs": diffs
            }

            st.session_state.comparison_history.append(comparison_entry)

    with col_btn2:
        if st.button("Clear Comparisons"):
            st.session_state.comparison_history = []

    # Show all saved comparisons
    if st.session_state.comparison_history:
        st.divider()
        st.subheader("📚 Comparison History with Visuals")

        for idx, comp in enumerate(st.session_state.comparison_history, start=1):
            main_x, main_y, main_z = comp["main_axes"]
            comp_x, comp_y, comp_z = comp["comp_axes"]
            diffs = comp["diffs"]

            st.markdown(f"## Comparison {idx}")

            c1, c2 = st.columns(2)

            with c1:
                fig_main = px.scatter_3d(
                    df,
                    x=main_x,
                    y=main_y,
                    z=main_z,
                    opacity=0.7,
                    title=f"Main View: {main_x} vs {main_y} vs {main_z}"
                )
                st.plotly_chart(fig_main, use_container_width=True)

            with c2:
                fig_comp = px.scatter_3d(
                    df,
                    x=comp_x,
                    y=comp_y,
                    z=comp_z,
                    opacity=0.7,
                    title=f"Compared View: {comp_x} vs {comp_y} vs {comp_z}"
                )
                st.plotly_chart(fig_comp, use_container_width=True)

            # Optional 2D support visuals
            st.markdown("### 2D Projections")
            p1, p2, p3 = st.columns(3)

            with p1:
                fig_xy = px.scatter(df, x=comp_x, y=comp_y, title=f"{comp_x} vs {comp_y}")
                st.plotly_chart(fig_xy, use_container_width=True)

            with p2:
                fig_xz = px.scatter(df, x=comp_x, y=comp_z, title=f"{comp_x} vs {comp_z}")
                st.plotly_chart(fig_xz, use_container_width=True)

            with p3:
                fig_yz = px.scatter(df, x=comp_y, y=comp_z, title=f"{comp_y} vs {comp_z}")
                st.plotly_chart(fig_yz, use_container_width=True)

            st.markdown("### Interpretation Differences")
            if diffs:
                for d in diffs:
                    st.markdown(f"- {d}")
            else:
                st.info("No major differences detected.")

            st.markdown("---")
