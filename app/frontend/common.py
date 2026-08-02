"""
Shared constants and helpers for the Dash frontend pages.

Kept UI-framework-agnostic where possible (no `dash.*`/`dcc.*` calls here)
so the analytics-fetching and chart-building logic stays reusable.
"""

import base64
import os

import cv2
import requests

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")
PLACEHOLDER_TEXT = "A comida estava deliciosa mas o serviço foi muito lento."

DARK_BG = "#16171a"
CARD_BG = "#212226"
BORDER_COLOR = "#33343a"
TEXT_COLOR = "#e6e6e6"
MUTED_COLOR = "#96979d"
ACCENT_COLOR = "#4c7cf0"

FONT_FAMILY = (
    "-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif"
)

CARD_STYLE = {
    "backgroundColor": CARD_BG,
    "borderRadius": "10px",
    "border": f"1px solid {BORDER_COLOR}",
    "padding": "20px",
}

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

CHART_LAYOUT = dict(
    paper_bgcolor=CARD_BG,
    plot_bgcolor=CARD_BG,
    font_color=TEXT_COLOR,
    margin=dict(l=40, r=20, t=40, b=40),
    legend=dict(bgcolor="rgba(0,0,0,0)"),
)


def get_json(path: str, default, params: dict | None = None):
    try:
        resp = requests.get(f"{BACKEND_URL}{path}", params=params, timeout=10)
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


def frame_to_data_uri(frame) -> str:
    """render_smiley() returns a BGRA numpy array; encode it as a PNG data URI for html.Img."""
    ok, buf = cv2.imencode(".png", frame)
    if not ok:
        raise RuntimeError("Falha ao codificar o smiley em PNG.")
    b64 = base64.b64encode(buf).decode("ascii")
    return f"data:image/png;base64,{b64}"


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
