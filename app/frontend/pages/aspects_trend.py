"""Aspetos & Tendência - leaderboard de aspetos mais criticados + evolução temporal."""

import plotly.graph_objects as go
import streamlit as st

from common import CHART_LAYOUT, POLARITY_COLORS, get_json

st.title("📈 Aspetos & Tendência")
if st.button("🔄 Atualizar"):
    st.rerun()

col1, col2 = st.columns(2)

with col1:
    st.subheader("Aspetos mais criticados")
    rows = sorted(get_json("/analytics/top-negative-aspects", []), key=lambda r: r["count"])
    fig = go.Figure(
        go.Bar(
            x=[r["count"] for r in rows],
            y=[r["aspect"] for r in rows],
            orientation="h",
            marker_color=POLARITY_COLORS["negative"],
        )
    )
    fig.update_layout(xaxis_title="Nº de menções negativas", **CHART_LAYOUT)
    st.plotly_chart(fig, width="stretch")

with col2:
    st.subheader("Sentimento ao longo do tempo")
    rows = get_json("/analytics/sentiment-over-time", [])
    by_polarity: dict[str, list[tuple[str, int]]] = {}
    for r in rows:
        pol = r.get("polarity") or "neutral"
        by_polarity.setdefault(pol, []).append((r["date"], r["count"]))

    fig2 = go.Figure()
    for polarity, color in POLARITY_COLORS.items():
        points = sorted(by_polarity.get(polarity, []))
        fig2.add_trace(
            go.Scatter(
                x=[p[0] for p in points],
                y=[p[1] for p in points],
                mode="lines+markers",
                name=polarity.capitalize(),
                line_color=color,
            )
        )
    fig2.update_layout(xaxis_title="Data", yaxis_title="Nº de menções", **CHART_LAYOUT)
    st.plotly_chart(fig2, width="stretch")
