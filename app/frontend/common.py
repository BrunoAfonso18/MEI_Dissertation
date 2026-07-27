"""
Shared constants and helpers for the Streamlit dashboard pages.

Kept framework-agnostic on purpose (no `st.*` calls here) so it can be
imported both by the entry point and by every page under pages/.
"""

import os

import requests

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")
PLACEHOLDER_TEXT = "A comida estava deliciosa mas o serviço foi muito lento."

POLARITY_COLORS = {"positive": "#4caf50", "neutral": "#f5d76e", "negative": "#e05555"}

CATEGORY_LABELS = {
    "FOOD_QUALITY": "Qualidade da Comida",
    "SERVICE": "Serviço",
    "PRICE_AND_VALUE": "Preço / Valor",
    "AMBIENCE_AND_ATMOSPHERE": "Ambiente",
    "LOCATION": "Localização",
    "PORTION_SIZE": "Tamanho da Dose",
    "MENU_VARIETY": "Variedade do Menu",
    "CLEANLINESS": "Limpeza",
    "WAITING_TIME": "Tempo de Espera",
}

# Transparent backgrounds so charts blend into Streamlit's own light/dark theme
# instead of hardcoding colors (unlike the previous Dash version, which had to
# paint its own dark theme from scratch).
CHART_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    margin=dict(l=40, r=20, t=40, b=40),
)


def get_json(path: str, default):
    try:
        resp = requests.get(f"{BACKEND_URL}{path}", timeout=10)
        resp.raise_for_status()
        return resp.json()
    except requests.RequestException:
        return default


def error_detail(exc: requests.RequestException) -> str:
    response = getattr(exc, "response", None)
    if response is not None:
        try:
            return response.json().get("detail", str(exc))
        except ValueError:
            pass
    return str(exc)


def fetch_restaurants() -> list[dict]:
    return get_json("/restaurants", [])


def to_rgba(frame):
    """render_smiley() returns BGRA; st.image wants RGB(A)."""
    return frame[:, :, [2, 1, 0, 3]]


def crisp_positivity(crisp_score) -> float:
    """Map an average fuzzy_crisp_score in [0, 1] to the [-1, 1] range render_smiley expects."""
    return (crisp_score - 0.5) * 2 if crisp_score is not None else 0.0


def pivot_category_counts(rows: list[dict]) -> dict:
    """category -> {polarity: count}, seeded with the known category taxonomy."""
    pivot = {cat: {"positive": 0, "neutral": 0, "negative": 0} for cat in CATEGORY_LABELS}
    for r in rows:
        cat = r.get("category")
        pol = r.get("polarity") or "neutral"
        bucket = pivot.setdefault(cat, {"positive": 0, "neutral": 0, "negative": 0})
        bucket[pol] = bucket.get(pol, 0) + r.get("count", 0)
    return pivot
