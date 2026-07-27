"""Categorias - sentimento por categoria (bar chart) e volume por categoria (radar)."""

import plotly.graph_objects as go
import streamlit as st

from common import CATEGORY_LABELS, CHART_LAYOUT, POLARITY_COLORS, get_json, pivot_category_counts

st.title("🏷️ Categorias de Aspeto")
if st.button("🔄 Atualizar"):
    st.rerun()

rows = get_json("/analytics/sentiment-by-category", [])
pivot = pivot_category_counts(rows)
cats = list(pivot.keys())
labels = [CATEGORY_LABELS.get(c, c) for c in cats]

col1, col2 = st.columns(2)

with col1:
    fig = go.Figure()
    for polarity, color in POLARITY_COLORS.items():
        fig.add_bar(
            name=polarity.capitalize(),
            x=labels,
            y=[pivot[c][polarity] for c in cats],
            marker_color=color,
        )
    fig.update_layout(barmode="stack", title="Sentimento por categoria de aspeto", **CHART_LAYOUT)
    st.plotly_chart(fig, width="stretch")

with col2:
    totals = [sum(pivot[c].values()) for c in cats]
    fig2 = go.Figure(
        go.Scatterpolar(r=totals + totals[:1], theta=labels + labels[:1], fill="toself", name="Nº de menções")
    )
    fig2.update_layout(
        polar=dict(radialaxis=dict(visible=True)),
        title="Volume de menções por categoria",
        showlegend=False,
        **CHART_LAYOUT,
    )
    st.plotly_chart(fig2, width="stretch")
