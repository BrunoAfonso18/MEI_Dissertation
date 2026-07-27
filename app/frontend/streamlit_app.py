"""
Streamlit entry point - ABSA sentiment smiley + review submission + dashboard.

Replaces the previous Plotly Dash frontend. Pages live under pages/ and are
wired into a sidebar via st.navigation, grouped into "Reviews" (submit a
single review or a bulk CSV - persisted already-processed into the data
warehouse) and "Dashboard" (read-only views over the data warehouse via the
backend's /analytics/* endpoints).

Run with:
    streamlit run app/frontend/streamlit_app.py
"""

import streamlit as st

st.set_page_config(page_title="ABSA Sentiment Dashboard", page_icon="🙂", layout="wide")

pages = {
    "Reviews": [
        st.Page("pages/submit_review.py", title="Submeter Review", icon="📝", default=True),
    ],
    "Dashboard": [
        st.Page("pages/overview.py", title="Visão Geral", icon="📊"),
        st.Page("pages/categories.py", title="Categorias", icon="🏷️"),
        st.Page("pages/aspects_trend.py", title="Aspetos & Tendência", icon="📈"),
        st.Page("pages/restaurants.py", title="Restaurantes", icon="🍽️"),
    ],
}

navigation = st.navigation(pages)
navigation.run()
