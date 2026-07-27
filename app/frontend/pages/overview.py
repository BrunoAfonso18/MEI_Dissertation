"""Visão Geral - KPI cards + a mini-smiley per aspect category."""

import streamlit as st

from common import CATEGORY_LABELS, crisp_positivity, get_json, to_rgba
from smiley import render_smiley

st.title("📊 Visão Geral")
if st.button("🔄 Atualizar"):
    st.rerun()

overview = get_json("/analytics/overview", {})
avg_score = overview.get("avg_crisp_score")

col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    st.image(to_rgba(render_smiley(crisp_positivity(avg_score), (70, 70))), width=70)
    st.metric("Sentimento geral", f"{avg_score:.2f}" if avg_score is not None else "—")
with col2:
    st.metric("Total de reviews", overview.get("total_reviews", 0))
with col3:
    st.metric("% Positivas", f"{overview.get('pct_positive', 0)}%")
with col4:
    st.metric("% Negativas", f"{overview.get('pct_negative', 0)}%")
with col5:
    st.metric("% Neutras", f"{overview.get('pct_neutral', 0)}%")

st.divider()
st.subheader("Sentimento por categoria de aspeto")

rows = get_json("/analytics/sentiment-by-category", [])
weighted = {cat: [0.0, 0] for cat in CATEGORY_LABELS}
for r in rows:
    cat = r.get("category")
    avg = r.get("avg_crisp_score")
    count = r.get("count", 0)
    if cat not in weighted or avg is None:
        continue
    weighted[cat][0] += avg * count
    weighted[cat][1] += count

cols = st.columns(len(CATEGORY_LABELS))
for col, (cat, label) in zip(cols, CATEGORY_LABELS.items()):
    total_score, total_count = weighted[cat]
    with col:
        if total_count > 0:
            score = total_score / total_count
            st.image(to_rgba(render_smiley(crisp_positivity(score), (90, 90))), width=90)
            st.caption(f"**{label}**  \n{score:.2f} ({total_count})")
        else:
            st.image(to_rgba(render_smiley(0.0, (90, 90))), width=90)
            st.caption(f"**{label}**  \nSem dados")
