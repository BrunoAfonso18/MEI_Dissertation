"""
Submeter Review - live smiley demo + single/bulk submission into the DW.

The ABSA + fuzzy pipeline runs locally, in-process, and reacts to whatever is
typed in the textbox on a ~300ms poll (dcc.Interval + dcc.Store), so the
smiley updates live, keystroke by keystroke - the same interaction the
desktop/Dash versions always had, restored here after the Streamlit
experiment (which could only rerun on blur/Ctrl+Enter).

Submitting a review or a CSV calls the backend (POST /query,
POST /reviews/upload), which persists the already-processed result straight
into the data warehouse.
"""

import base64

import dash
import requests
from dash import Input, Output, State, callback, dcc, html
from dash.exceptions import PreventUpdate

from absa_pipeline import AbsaSentimentPipeline
from common import (
    ACCENT_COLOR,
    BACKEND_URL,
    CARD_STYLE,
    PLACEHOLDER_TEXT,
    TEXT_COLOR,
    error_detail,
    fetch_restaurants,
    frame_to_data_uri,
)
from smiley import render_smiley

dash.register_page(__name__, path="/", name="Submeter Review", category="Reviews", order=0)

SMILEY_SIZE = (280, 280)
DEBOUNCE_MS = 300

pipeline = AbsaSentimentPipeline()


def layout():
    restaurant_options = [
        {"label": f"{r['name']} ({r.get('district') or '—'})", "value": r["id_restaurant"]}
        for r in fetch_restaurants()
    ]
    initial_frame = render_smiley(0.0, SMILEY_SIZE)

    return html.Div(
        [
            html.H2("ABSA Sentiment Smiley"),
            html.Div(
                className="app-card",
                style={**CARD_STYLE, "maxWidth": "700px", "marginBottom": "24px"},
                children=[
                    html.Label("Escreve uma review de restaurante:", style={"fontWeight": "bold"}),
                    dcc.Textarea(
                        id="review-text",
                        value=PLACEHOLDER_TEXT,
                        style={
                            "width": "100%",
                            "height": "100px",
                            "marginTop": "8px",
                            "backgroundColor": "#333",
                            "color": TEXT_COLOR,
                            "border": "none",
                            "borderRadius": "6px",
                            "padding": "10px",
                            "fontSize": "14px",
                        },
                    ),
                    html.Div(
                        style={"display": "flex", "justifyContent": "center", "margin": "16px 0"},
                        children=[
                            html.Img(
                                id="smiley-image",
                                src=frame_to_data_uri(initial_frame),
                                style={"width": f"{SMILEY_SIZE[0]}px", "height": f"{SMILEY_SIZE[1]}px"},
                            )
                        ],
                    ),
                    html.Div(
                        id="smiley-status",
                        children="A carregar o modelo...",
                        style={"textAlign": "center", "fontWeight": "bold", "marginBottom": "10px"},
                    ),
                    html.Pre(
                        id="aspects-box",
                        style={
                            "backgroundColor": "#111",
                            "padding": "10px",
                            "borderRadius": "6px",
                            "minHeight": "80px",
                            "fontFamily": "Consolas, monospace",
                            "fontSize": "12px",
                            "whiteSpace": "pre-wrap",
                        },
                    ),
                    html.Hr(),
                    html.Label("Restaurante:", style={"fontWeight": "bold"}),
                    dcc.Dropdown(
                        id="restaurant-dropdown",
                        options=restaurant_options,
                        value=restaurant_options[0]["value"] if restaurant_options else None,
                        style={"color": "#000", "marginTop": "8px", "marginBottom": "12px"},
                    ),
                    html.Button(
                        "Submeter review",
                        id="submit-button",
                        n_clicks=0,
                        style={
                            "padding": "10px 20px",
                            "borderRadius": "6px",
                            "border": "none",
                            "backgroundColor": ACCENT_COLOR,
                            "color": "white",
                            "fontWeight": "600",
                            "cursor": "pointer",
                        },
                    ),
                    html.Div(id="submit-status", style={"marginTop": "12px"}),
                ],
            ),
            html.Div(
                className="app-card",
                style={**CARD_STYLE, "maxWidth": "700px"},
                children=[
                    html.H4("Submissão em bulk (CSV)"),
                    html.P(
                        "Colunas esperadas: restaurant_id, text",
                        style={"color": "#999", "fontSize": "13px"},
                    ),
                    dcc.Upload(
                        id="bulk-upload",
                        children=html.Div(["Arrasta um ficheiro CSV ou ", html.A("seleciona um")]),
                        style={
                            "width": "100%",
                            "height": "60px",
                            "lineHeight": "60px",
                            "borderWidth": "1px",
                            "borderStyle": "dashed",
                            "borderRadius": "6px",
                            "textAlign": "center",
                            "borderColor": "#555",
                        },
                        multiple=False,
                    ),
                    html.Div(id="bulk-status", style={"marginTop": "12px"}),
                ],
            ),
            dcc.Store(id="last-processed-text", data=""),
            dcc.Interval(id="smiley-interval", interval=DEBOUNCE_MS, n_intervals=0),
        ]
    )


@callback(
    Output("smiley-image", "src"),
    Output("smiley-status", "children"),
    Output("aspects-box", "children"),
    Output("last-processed-text", "data"),
    Input("smiley-interval", "n_intervals"),
    State("review-text", "value"),
    State("last-processed-text", "data"),
)
def update_smiley(_n_intervals, text, last_text):
    text = (text or "").strip()
    if text == last_text:
        raise PreventUpdate

    result = pipeline.run(text)
    src = frame_to_data_uri(render_smiley(result.positivity, SMILEY_SIZE))

    if result.aspects:
        status = f"Sentimento geral: {result.positivity:+.2f}  ({result.label})"
    else:
        status = "Nenhum aspeto detetado - smiley neutro"

    aspects_text = "\n".join(
        f"[{a.polarity:<8}] {a.term:<25} confiança={a.confidence:.2f}" for a in result.aspects
    )

    return src, status, aspects_text, text


@callback(
    Output("submit-status", "children"),
    Output("review-text", "value"),
    Input("submit-button", "n_clicks"),
    State("review-text", "value"),
    State("restaurant-dropdown", "value"),
    prevent_initial_call=True,
)
def submit_review(_n_clicks, text, restaurant_id):
    text = (text or "").strip()
    if not text:
        return html.Div("Escreve uma review antes de submeter.", style={"color": "#e07b39"}), dash.no_update
    if restaurant_id is None:
        return html.Div("Seleciona um restaurante antes de submeter.", style={"color": "#e07b39"}), dash.no_update

    try:
        resp = requests.post(
            f"{BACKEND_URL}/query",
            json={"text": text, "restaurant_id": restaurant_id},
            timeout=30,
        )
        resp.raise_for_status()
    except requests.RequestException as e:
        return html.Div(f"Erro ao submeter: {error_detail(e)}", style={"color": "#e05555"}), dash.no_update

    data = resp.json()
    n_aspects = len(data.get("aspects", []))
    return (
        html.Div(
            f"Review #{data['id']} guardada no data warehouse com {n_aspects} aspeto(s) processado(s).",
            style={"color": "#4caf50"},
        ),
        "",
    )


@callback(
    Output("bulk-status", "children"),
    Input("bulk-upload", "contents"),
    State("bulk-upload", "filename"),
    prevent_initial_call=True,
)
def submit_bulk(contents, filename):
    if contents is None:
        raise PreventUpdate
    if not filename.lower().endswith(".csv"):
        return html.Div("O ficheiro tem de ser um CSV.", style={"color": "#e07b39"})

    _header, b64_data = contents.split(",", 1)
    file_bytes = base64.b64decode(b64_data)

    try:
        resp = requests.post(
            f"{BACKEND_URL}/reviews/upload",
            files={"file": (filename, file_bytes, "text/csv")},
            timeout=120,
        )
        resp.raise_for_status()
    except requests.RequestException as e:
        return html.Div(f"Erro ao processar o ficheiro: {error_detail(e)}", style={"color": "#e05555"})

    data = resp.json()
    return html.Div(
        [
            html.Div(
                f"Total: {data['total']}  |  Processadas: {data['processed']}  |  Falhadas: {data['failed']}",
                style={"color": "#4caf50" if data["failed"] == 0 else "#e07b39"},
            ),
            html.Ul([html.Li(err) for err in data["errors"]]) if data.get("errors") else None,
        ]
    )
