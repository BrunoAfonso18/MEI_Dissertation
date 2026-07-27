"""Restaurantes - ranking (tabela) e score fuzzy médio vs volume de reviews (scatter)."""

import plotly.graph_objects as go
import streamlit as st

from common import CHART_LAYOUT, get_json

st.title("🍽️ Restaurantes")
if st.button("🔄 Atualizar"):
    st.rerun()

rows = get_json("/analytics/restaurant-performance", [])

col1, col2 = st.columns(2)

with col1:
    st.subheader("Ranking de restaurantes")
    if not rows:
        st.info("Ainda sem reviews associadas a restaurantes.")
    else:
        table_rows = [
            {
                "Restaurante": r["name"],
                "Distrito": r.get("district") or "—",
                "Score médio": round(r["avg_crisp_score"], 2) if r["avg_crisp_score"] is not None else None,
                "Nº reviews": r["review_count"],
            }
            for r in rows
        ]
        st.dataframe(table_rows, width="stretch", hide_index=True)

with col2:
    st.subheader("Score fuzzy médio vs volume de reviews")
    plot_rows = [r for r in rows if r.get("avg_crisp_score") is not None]
    fig = go.Figure(
        go.Scatter(
            x=[r["review_count"] for r in plot_rows],
            y=[r["avg_crisp_score"] for r in plot_rows],
            mode="markers+text",
            text=[r["name"] for r in plot_rows],
            textposition="top center",
            marker=dict(size=14, color=[r["avg_crisp_score"] for r in plot_rows], colorscale="RdYlGn", cmin=0, cmax=1),
        )
    )
    fig.update_layout(xaxis_title="Nº de reviews", yaxis_title="Score fuzzy médio", **CHART_LAYOUT)
    st.plotly_chart(fig, width="stretch")
