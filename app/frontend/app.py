"""
Plotly Dash frontend - ABSA sentiment smiley + review submission + dashboard.

The smiley keeps reacting locally, in-process, to whatever is typed in the
textbox (same pipeline and same ~300ms polling cadence as the previous
customtkinter desktop app), so its look and feel is unchanged. It lives in the
"Submeter Review" tab together with the review submission and bulk-CSV-upload
flows, exactly as before.

The "Dashboard" tab is new: it visualizes the data warehouse (via the
backend's /analytics/* endpoints) with per-aspect-category smileys, KPI
cards, and charts (categories, top negative aspects, sentiment over time,
restaurant ranking/scatter). Data is only fetched when that tab is opened or
the "Atualizar dashboard" button is clicked - it does not poll.

Run with:
    python app/frontend/app.py
"""

import base64
import os

import cv2
import plotly.graph_objects as go
import requests
from dash import Dash, Input, Output, State, dash_table, dcc, html
from dash.exceptions import PreventUpdate

from absa_pipeline import AbsaSentimentPipeline
from smiley import render_smiley

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")
PLACEHOLDER_TEXT = "A comida estava deliciosa mas o serviço foi muito lento."
SMILEY_SIZE = (320, 320)
CATEGORY_SMILEY_SIZE = (90, 90)
KPI_SMILEY_SIZE = (60, 60)
DEBOUNCE_MS = 300

DARK_BG = "#1a1a1a"
CARD_BG = "#242424"
TEXT_COLOR = "#e6e6e6"
MUTED_COLOR = "#999"

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

pipeline = AbsaSentimentPipeline()


# ── Helpers ──────────────────────────────────────────────────────

def _frame_to_data_uri(frame) -> str:
    ok, buf = cv2.imencode(".png", frame)
    if not ok:
        raise RuntimeError("Falha ao codificar o smiley em PNG.")
    b64 = base64.b64encode(buf).decode("ascii")
    return f"data:image/png;base64,{b64}"


def _smiley_data_uri(crisp_score, size=CATEGORY_SMILEY_SIZE) -> str:
    positivity = (crisp_score - 0.5) * 2 if crisp_score is not None else 0.0
    return _frame_to_data_uri(render_smiley(positivity, size))


def _get_json(path: str, default):
    try:
        resp = requests.get(f"{BACKEND_URL}{path}", timeout=10)
        resp.raise_for_status()
        return resp.json()
    except requests.RequestException:
        return default


def _error_detail(exc: requests.RequestException) -> str:
    response = getattr(exc, "response", None)
    if response is not None:
        try:
            return response.json().get("detail", str(exc))
        except ValueError:
            pass
    return str(exc)


def _fetch_restaurants() -> list[dict]:
    return _get_json("/restaurants", [])


def _pivot_category_counts(rows: list[dict]) -> dict:
    """category -> {polarity: count}, seeded with the known category taxonomy."""
    pivot = {cat: {"positive": 0, "neutral": 0, "negative": 0} for cat in CATEGORY_LABELS}
    for r in rows:
        cat = r.get("category")
        pol = r.get("polarity") or "neutral"
        bucket = pivot.setdefault(cat, {"positive": 0, "neutral": 0, "negative": 0})
        bucket[pol] = bucket.get(pol, 0) + r.get("count", 0)
    return pivot


def _card(label: str, value) -> html.Div:
    return html.Div(
        style={
            "backgroundColor": CARD_BG,
            "borderRadius": "10px",
            "padding": "16px 20px",
            "minWidth": "140px",
        },
        children=[
            html.Div(label, style={"fontSize": "12px", "color": MUTED_COLOR}),
            html.Div(str(value), style={"fontSize": "22px", "fontWeight": "bold"}),
        ],
    )


def _chart_card(children) -> html.Div:
    return html.Div(
        style={
            "flex": "1",
            "minWidth": "420px",
            "backgroundColor": CARD_BG,
            "borderRadius": "10px",
            "padding": "16px",
        },
        children=children,
    )


# ── App & layout ─────────────────────────────────────────────────

app = Dash(__name__)
app.title = "ABSA Sentiment Smiley"

TAB_STYLE = {"backgroundColor": CARD_BG, "color": TEXT_COLOR, "border": "1px solid #333", "padding": "10px"}
TAB_SELECTED_STYLE = {"backgroundColor": "#2f6fed", "color": "white", "border": "1px solid #2f6fed", "padding": "10px"}


def build_submit_tab(restaurant_options: list[dict]):
    initial_frame = render_smiley(0.0, SMILEY_SIZE)
    return html.Div(
        style={"paddingTop": "20px"},
        children=[
            html.Div(
                style={
                    "backgroundColor": CARD_BG,
                    "borderRadius": "10px",
                    "padding": "20px",
                    "maxWidth": "700px",
                    "marginBottom": "24px",
                },
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
                                src=_frame_to_data_uri(initial_frame),
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
                            "backgroundColor": "#2f6fed",
                            "color": "white",
                            "cursor": "pointer",
                        },
                    ),
                    html.Div(id="submit-status", style={"marginTop": "12px"}),
                ],
            ),
            html.Div(
                style={
                    "backgroundColor": CARD_BG,
                    "borderRadius": "10px",
                    "padding": "20px",
                    "maxWidth": "700px",
                },
                children=[
                    html.H4("Submissão em bulk (CSV)"),
                    html.P(
                        "Colunas esperadas: restaurant_id, text",
                        style={"color": MUTED_COLOR, "fontSize": "13px"},
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
        ],
    )


def build_dashboard_tab():
    return html.Div(
        style={"paddingTop": "20px", "maxWidth": "1100px"},
        children=[
            html.Div(
                id="kpi-row",
                style={"display": "flex", "gap": "16px", "flexWrap": "wrap", "marginBottom": "24px"},
            ),
            html.Div(
                style={
                    "backgroundColor": CARD_BG,
                    "borderRadius": "10px",
                    "padding": "20px",
                    "marginBottom": "24px",
                },
                children=[
                    html.H4("Sentimento por categoria de aspeto"),
                    html.Div(
                        id="category-smiley-grid",
                        style={"display": "flex", "flexWrap": "wrap", "gap": "16px"},
                    ),
                ],
            ),
            html.Div(
                style={"display": "flex", "gap": "20px", "flexWrap": "wrap", "marginBottom": "24px"},
                children=[
                    _chart_card(dcc.Graph(id="category-bar-chart")),
                    _chart_card(dcc.Graph(id="category-radar-chart")),
                ],
            ),
            html.Div(
                style={"display": "flex", "gap": "20px", "flexWrap": "wrap", "marginBottom": "24px"},
                children=[
                    _chart_card(dcc.Graph(id="top-negative-chart")),
                    _chart_card(dcc.Graph(id="time-trend-chart")),
                ],
            ),
            html.Div(
                style={"display": "flex", "gap": "20px", "flexWrap": "wrap", "marginBottom": "24px"},
                children=[
                    _chart_card([html.H4("Ranking de restaurantes"), html.Div(id="restaurant-table-container")]),
                    _chart_card(dcc.Graph(id="restaurant-scatter-chart")),
                ],
            ),
            html.Button(
                "🔄 Atualizar dashboard",
                id="dashboard-refresh",
                n_clicks=0,
                style={
                    "padding": "10px 20px",
                    "borderRadius": "6px",
                    "border": "none",
                    "backgroundColor": "#2f6fed",
                    "color": "white",
                    "cursor": "pointer",
                    "marginBottom": "20px",
                },
            ),
            dcc.Store(id="category-data"),
            dcc.Store(id="restaurant-data"),
        ],
    )


def serve_layout():
    restaurant_options = [
        {"label": f"{r['name']} ({r.get('district') or '—'})", "value": r["id_restaurant"]}
        for r in _fetch_restaurants()
    ]

    return html.Div(
        style={
            "backgroundColor": DARK_BG,
            "color": TEXT_COLOR,
            "minHeight": "100vh",
            "fontFamily": "Segoe UI, sans-serif",
            "padding": "30px",
        },
        children=[
            html.H2("ABSA Sentiment Smiley"),
            dcc.Tabs(
                id="main-tabs",
                value="tab-submit",
                children=[
                    dcc.Tab(
                        label="Submeter Review",
                        value="tab-submit",
                        style=TAB_STYLE,
                        selected_style=TAB_SELECTED_STYLE,
                        children=[build_submit_tab(restaurant_options)],
                    ),
                    dcc.Tab(
                        label="Dashboard",
                        value="tab-dashboard",
                        style=TAB_STYLE,
                        selected_style=TAB_SELECTED_STYLE,
                        children=[build_dashboard_tab()],
                    ),
                ],
            ),
        ],
    )


app.layout = serve_layout


# ── "Submeter Review" tab callbacks (unchanged behaviour) ─────────

@app.callback(
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
    src = _frame_to_data_uri(render_smiley(result.positivity, SMILEY_SIZE))

    if result.aspects:
        status = f"Sentimento geral: {result.positivity:+.2f}  ({result.label})"
    else:
        status = "Nenhum aspeto detetado - smiley neutro"

    aspects_text = "\n".join(
        f"[{a.polarity:<8}] {a.term:<25} confiança={a.confidence:.2f}" for a in result.aspects
    )

    return src, status, aspects_text, text


@app.callback(
    Output("submit-status", "children"),
    Input("submit-button", "n_clicks"),
    State("review-text", "value"),
    State("restaurant-dropdown", "value"),
    prevent_initial_call=True,
)
def submit_review(_n_clicks, text, restaurant_id):
    text = (text or "").strip()
    if not text:
        return html.Div("Escreve uma review antes de submeter.", style={"color": "#e07b39"})
    if restaurant_id is None:
        return html.Div("Seleciona um restaurante antes de submeter.", style={"color": "#e07b39"})

    try:
        resp = requests.post(
            f"{BACKEND_URL}/query",
            json={"text": text, "restaurant_id": restaurant_id},
            timeout=30,
        )
        resp.raise_for_status()
    except requests.RequestException as e:
        return html.Div(f"Erro ao submeter: {_error_detail(e)}", style={"color": "#e05555"})

    data = resp.json()
    n_aspects = len(data.get("aspects", []))
    return html.Div(
        f"Review #{data['id']} guardada no data warehouse com {n_aspects} aspeto(s) processado(s).",
        style={"color": "#4caf50"},
    )


@app.callback(
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
        return html.Div(f"Erro ao processar o ficheiro: {_error_detail(e)}", style={"color": "#e05555"})

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


# ── "Dashboard" tab: lazy data loading ────────────────────────────
# Fetched only when the Dashboard tab is opened or "Atualizar dashboard" is
# clicked - not polled, since the underlying data only changes when someone
# submits a review.

@app.callback(
    Output("category-data", "data"),
    Input("main-tabs", "value"),
    Input("dashboard-refresh", "n_clicks"),
)
def load_category_data(tab_value, _n_clicks):
    if tab_value != "tab-dashboard":
        raise PreventUpdate
    return _get_json("/analytics/sentiment-by-category", [])


@app.callback(
    Output("restaurant-data", "data"),
    Input("main-tabs", "value"),
    Input("dashboard-refresh", "n_clicks"),
)
def load_restaurant_data(tab_value, _n_clicks):
    if tab_value != "tab-dashboard":
        raise PreventUpdate
    return _get_json("/analytics/restaurant-performance", [])


@app.callback(
    Output("kpi-row", "children"),
    Input("main-tabs", "value"),
    Input("dashboard-refresh", "n_clicks"),
)
def render_kpis(tab_value, _n_clicks):
    if tab_value != "tab-dashboard":
        raise PreventUpdate

    data = _get_json("/analytics/overview", {})
    avg_score = data.get("avg_crisp_score")

    return [
        html.Div(
            style={
                "backgroundColor": CARD_BG,
                "borderRadius": "10px",
                "padding": "10px 20px",
                "display": "flex",
                "alignItems": "center",
                "gap": "10px",
            },
            children=[
                html.Img(
                    src=_smiley_data_uri(avg_score, KPI_SMILEY_SIZE),
                    style={"width": f"{KPI_SMILEY_SIZE[0]}px", "height": f"{KPI_SMILEY_SIZE[1]}px"},
                ),
                html.Div(
                    [
                        html.Div("Sentimento geral", style={"fontSize": "12px", "color": MUTED_COLOR}),
                        html.Div(
                            f"{avg_score:.2f}" if avg_score is not None else "—",
                            style={"fontSize": "20px", "fontWeight": "bold"},
                        ),
                    ]
                ),
            ],
        ),
        _card("Total de reviews", data.get("total_reviews", 0)),
        _card("% Positivas", f"{data.get('pct_positive', 0)}%"),
        _card("% Negativas", f"{data.get('pct_negative', 0)}%"),
        _card("% Neutras", f"{data.get('pct_neutral', 0)}%"),
    ]


@app.callback(
    Output("category-smiley-grid", "children"),
    Input("category-data", "data"),
)
def render_category_smileys(rows):
    if rows is None:
        raise PreventUpdate

    weighted = {cat: [0.0, 0] for cat in CATEGORY_LABELS}
    for r in rows:
        cat = r.get("category")
        avg = r.get("avg_crisp_score")
        count = r.get("count", 0)
        if cat not in weighted or avg is None:
            continue
        weighted[cat][0] += avg * count
        weighted[cat][1] += count

    tiles = []
    for cat, label in CATEGORY_LABELS.items():
        total_score, total_count = weighted[cat]
        if total_count > 0:
            score = total_score / total_count
            subtitle = f"{score:.2f}  ({total_count})"
        else:
            score = None
            subtitle = "Sem dados"
        tiles.append(
            html.Div(
                style={"textAlign": "center", "width": "110px"},
                children=[
                    html.Img(
                        src=_smiley_data_uri(score, CATEGORY_SMILEY_SIZE),
                        style={"width": f"{CATEGORY_SMILEY_SIZE[0]}px", "height": f"{CATEGORY_SMILEY_SIZE[1]}px"},
                    ),
                    html.Div(label, style={"fontSize": "12px", "fontWeight": "bold", "marginTop": "4px"}),
                    html.Div(subtitle, style={"fontSize": "11px", "color": MUTED_COLOR}),
                ],
            )
        )
    return tiles


@app.callback(
    Output("category-bar-chart", "figure"),
    Input("category-data", "data"),
)
def render_category_bar(rows):
    if rows is None:
        raise PreventUpdate

    pivot = _pivot_category_counts(rows)
    cats = list(pivot.keys())
    labels = [CATEGORY_LABELS.get(c, c) for c in cats]

    fig = go.Figure()
    for polarity, color in POLARITY_COLORS.items():
        fig.add_bar(
            name=polarity.capitalize(),
            x=labels,
            y=[pivot[c][polarity] for c in cats],
            marker_color=color,
        )
    fig.update_layout(barmode="stack", title="Sentimento por categoria de aspeto", **CHART_LAYOUT)
    return fig


@app.callback(
    Output("category-radar-chart", "figure"),
    Input("category-data", "data"),
)
def render_category_radar(rows):
    if rows is None:
        raise PreventUpdate

    pivot = _pivot_category_counts(rows)
    cats = list(pivot.keys())
    labels = [CATEGORY_LABELS.get(c, c) for c in cats]
    totals = [sum(pivot[c].values()) for c in cats]

    fig = go.Figure(
        go.Scatterpolar(r=totals + totals[:1], theta=labels + labels[:1], fill="toself", name="Nº de menções")
    )
    fig.update_layout(
        polar=dict(bgcolor=CARD_BG, radialaxis=dict(visible=True, color=TEXT_COLOR), angularaxis=dict(color=TEXT_COLOR)),
        title="Volume de menções por categoria",
        showlegend=False,
        **CHART_LAYOUT,
    )
    return fig


@app.callback(
    Output("top-negative-chart", "figure"),
    Input("main-tabs", "value"),
    Input("dashboard-refresh", "n_clicks"),
)
def render_top_negative(tab_value, _n_clicks):
    if tab_value != "tab-dashboard":
        raise PreventUpdate

    rows = sorted(_get_json("/analytics/top-negative-aspects", []), key=lambda r: r["count"])
    fig = go.Figure(
        go.Bar(
            x=[r["count"] for r in rows],
            y=[r["aspect"] for r in rows],
            orientation="h",
            marker_color=POLARITY_COLORS["negative"],
        )
    )
    fig.update_layout(title="Aspetos mais criticados", xaxis_title="Nº de menções negativas", **CHART_LAYOUT)
    return fig


@app.callback(
    Output("time-trend-chart", "figure"),
    Input("main-tabs", "value"),
    Input("dashboard-refresh", "n_clicks"),
)
def render_time_trend(tab_value, _n_clicks):
    if tab_value != "tab-dashboard":
        raise PreventUpdate

    rows = _get_json("/analytics/sentiment-over-time", [])
    by_polarity: dict[str, list[tuple[str, int]]] = {}
    for r in rows:
        pol = r.get("polarity") or "neutral"
        by_polarity.setdefault(pol, []).append((r["date"], r["count"]))

    fig = go.Figure()
    for polarity, color in POLARITY_COLORS.items():
        points = sorted(by_polarity.get(polarity, []))
        fig.add_trace(
            go.Scatter(
                x=[p[0] for p in points],
                y=[p[1] for p in points],
                mode="lines+markers",
                name=polarity.capitalize(),
                line_color=color,
            )
        )
    fig.update_layout(title="Sentimento ao longo do tempo", xaxis_title="Data", yaxis_title="Nº de menções", **CHART_LAYOUT)
    return fig


@app.callback(
    Output("restaurant-table-container", "children"),
    Input("restaurant-data", "data"),
)
def render_restaurant_table(rows):
    if rows is None:
        raise PreventUpdate
    if not rows:
        return html.Div("Ainda sem reviews associadas a restaurantes.", style={"color": MUTED_COLOR})

    table_rows = [
        {
            "name": r["name"],
            "district": r.get("district") or "—",
            "avg_crisp_score": round(r["avg_crisp_score"], 2) if r["avg_crisp_score"] is not None else None,
            "review_count": r["review_count"],
        }
        for r in rows
    ]
    return dash_table.DataTable(
        columns=[
            {"name": "Restaurante", "id": "name"},
            {"name": "Distrito", "id": "district"},
            {"name": "Score médio", "id": "avg_crisp_score"},
            {"name": "Nº reviews", "id": "review_count"},
        ],
        data=table_rows,
        style_header={"backgroundColor": "#111", "color": TEXT_COLOR, "fontWeight": "bold"},
        style_cell={"backgroundColor": CARD_BG, "color": TEXT_COLOR, "border": "1px solid #333", "padding": "6px"},
        style_as_list_view=True,
        sort_action="native",
        page_size=8,
    )


@app.callback(
    Output("restaurant-scatter-chart", "figure"),
    Input("restaurant-data", "data"),
)
def render_restaurant_scatter(rows):
    if rows is None:
        raise PreventUpdate

    rows = [r for r in rows if r.get("avg_crisp_score") is not None]
    fig = go.Figure(
        go.Scatter(
            x=[r["review_count"] for r in rows],
            y=[r["avg_crisp_score"] for r in rows],
            mode="markers+text",
            text=[r["name"] for r in rows],
            textposition="top center",
            textfont=dict(color=TEXT_COLOR, size=10),
            marker=dict(size=14, color=[r["avg_crisp_score"] for r in rows], colorscale="RdYlGn", cmin=0, cmax=1),
        )
    )
    fig.update_layout(
        title="Score fuzzy médio vs volume de reviews",
        xaxis_title="Nº de reviews",
        yaxis_title="Score fuzzy médio",
        **CHART_LAYOUT,
    )
    return fig


def main() -> None:
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "8050")), debug=False)


if __name__ == "__main__":
    main()
