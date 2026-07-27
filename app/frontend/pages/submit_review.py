"""
Submeter Review - live smiley demo + single/bulk submission into the DW.

The ABSA + fuzzy pipeline runs locally, in-process (cached across reruns via
st.cache_resource so the model only loads once per server), exactly like the
previous Dash and customtkinter frontends. Streamlit reruns this script on
every widget interaction, so the smiley updates whenever the review text box
loses focus or you press Ctrl+Enter - not on every keystroke like the Dash
version's 300ms poll, since Streamlit has no push-based UI update model.

Submitting a review or a CSV calls the backend (POST /query,
POST /reviews/upload), which persists the already-processed result straight
into the data warehouse.
"""

import requests
import streamlit as st

from absa_pipeline import AbsaSentimentPipeline
from common import BACKEND_URL, PLACEHOLDER_TEXT, error_detail, fetch_restaurants, to_rgba
from smiley import render_smiley


@st.cache_resource
def _load_pipeline() -> AbsaSentimentPipeline:
    return AbsaSentimentPipeline()


st.title("😊 ABSA Sentiment Smiley")

pipeline = _load_pipeline()

text = st.text_area("Escreve uma review de restaurante:", value=PLACEHOLDER_TEXT, height=120)
result = pipeline.run(text.strip())

col_smiley, col_info = st.columns([1, 2])
with col_smiley:
    st.image(to_rgba(render_smiley(result.positivity, (280, 280))), width=280)
with col_info:
    if result.aspects:
        st.subheader(f"Sentimento geral: {result.positivity:+.2f}  ({result.label})")
        for aspect in result.aspects:
            st.text(f"[{aspect.polarity:<8}] {aspect.term:<25} confiança={aspect.confidence:.2f}")
    else:
        st.subheader("Nenhum aspeto detetado - smiley neutro")

st.divider()
st.subheader("Submeter review")

restaurants = fetch_restaurants()
options = {f"{r['name']} ({r.get('district') or '—'})": r["id_restaurant"] for r in restaurants}

if options:
    choice = st.selectbox("Restaurante:", list(options.keys()))
    restaurant_id = options[choice]
else:
    st.warning("Sem restaurantes disponíveis no backend.")
    restaurant_id = None

if st.button("Submeter review", type="primary"):
    if not text.strip():
        st.warning("Escreve uma review antes de submeter.")
    elif restaurant_id is None:
        st.warning("Seleciona um restaurante antes de submeter.")
    else:
        try:
            resp = requests.post(
                f"{BACKEND_URL}/query",
                json={"text": text.strip(), "restaurant_id": restaurant_id},
                timeout=30,
            )
            resp.raise_for_status()
            data = resp.json()
            st.success(
                f"Review #{data['id']} guardada no data warehouse com "
                f"{len(data.get('aspects', []))} aspeto(s) processado(s)."
            )
        except requests.RequestException as e:
            st.error(f"Erro ao submeter: {error_detail(e)}")

st.divider()
st.subheader("Submissão em bulk (CSV)")
st.caption("Colunas esperadas: restaurant_id, text")

uploaded = st.file_uploader("Carrega um ficheiro CSV", type=["csv"])
if uploaded is not None and st.button("Processar ficheiro"):
    try:
        resp = requests.post(
            f"{BACKEND_URL}/reviews/upload",
            files={"file": (uploaded.name, uploaded.getvalue(), "text/csv")},
            timeout=120,
        )
        resp.raise_for_status()
        data = resp.json()
        summary = f"Total: {data['total']}  |  Processadas: {data['processed']}  |  Falhadas: {data['failed']}"
        if data["failed"] == 0:
            st.success(summary)
        else:
            st.warning(summary)
        for err in data.get("errors", []):
            st.text(err)
    except requests.RequestException as e:
        st.error(f"Erro ao processar o ficheiro: {error_detail(e)}")
