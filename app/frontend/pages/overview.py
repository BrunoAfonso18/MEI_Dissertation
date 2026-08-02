"""Visão Geral - KPI cards + a mini-smiley per aspect category."""

import dash
from dash import Input, Output, callback, html

from common import CARD_STYLE, CATEGORY_LABELS, MUTED_COLOR, crisp_positivity, fetch_restaurants, frame_to_data_uri, get_json
from filters import filter_bar, filter_params
from smiley import render_smiley

dash.register_page(__name__, path="/visao-geral", name="Visão Geral", category="Dashboard", order=1)

KPI_SMILEY_SIZE = (60, 60)
CATEGORY_SMILEY_SIZE = (90, 90)


def _card(label: str, value) -> html.Div:
    return html.Div(
        className="app-card",
        style={**CARD_STYLE, "padding": "16px 20px", "minWidth": "140px"},
        children=[
            html.Div(label, style={"fontSize": "12px", "color": MUTED_COLOR}),
            html.Div(str(value), style={"fontSize": "22px", "fontWeight": "bold"}),
        ],
    )


def layout():
    return html.Div(
        [
            html.H2("Visão Geral"),
            filter_bar("overview", fetch_restaurants()),
            html.Div(id="overview-content"),
        ]
    )


@callback(
    Output("overview-content", "children"),
    Input("overview-district", "value"),
    Input("overview-restaurant", "value"),
    Input("overview-date-range", "start_date"),
    Input("overview-date-range", "end_date"),
)
def render_overview(district, restaurant_id, start_date, end_date):
    params = filter_params(district, restaurant_id, start_date, end_date)

    overview = get_json("/analytics/overview", {}, params=params)
    avg_score = overview.get("avg_crisp_score")

    rows = get_json("/analytics/sentiment-by-category", [], params=params)
    weighted = {cat: [0.0, 0] for cat in CATEGORY_LABELS}
    for r in rows:
        cat = r.get("category")
        avg = r.get("avg_crisp_score")
        count = r.get("count", 0)
        if cat not in weighted or avg is None:
            continue
        weighted[cat][0] += avg * count
        weighted[cat][1] += count

    category_tiles = []
    for cat, label in CATEGORY_LABELS.items():
        total_score, total_count = weighted[cat]
        if total_count > 0:
            score = total_score / total_count
            subtitle = f"{score:.2f} ({total_count})"
        else:
            score = None
            subtitle = "Sem dados"
        category_tiles.append(
            html.Div(
                style={"textAlign": "center", "width": "110px"},
                children=[
                    html.Img(
                        src=frame_to_data_uri(render_smiley(crisp_positivity(score), CATEGORY_SMILEY_SIZE)),
                        style={"width": f"{CATEGORY_SMILEY_SIZE[0]}px", "height": f"{CATEGORY_SMILEY_SIZE[1]}px"},
                    ),
                    html.Div(label, style={"fontSize": "12px", "fontWeight": "bold", "marginTop": "4px"}),
                    html.Div(subtitle, style={"fontSize": "11px", "color": MUTED_COLOR}),
                ],
            )
        )

    return html.Div(
        [
            html.Div(
                style={"display": "flex", "gap": "16px", "flexWrap": "wrap", "marginBottom": "24px"},
                children=[
                    html.Div(
                        className="app-card",
                        style={
                            **CARD_STYLE,
                            "padding": "10px 20px",
                            "display": "flex",
                            "alignItems": "center",
                            "gap": "10px",
                        },
                        children=[
                            html.Img(
                                src=frame_to_data_uri(render_smiley(crisp_positivity(avg_score), KPI_SMILEY_SIZE)),
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
                    _card("Total de reviews", overview.get("total_reviews", 0)),
                    _card("% Positivas", f"{overview.get('pct_positive', 0)}%"),
                    _card("% Negativas", f"{overview.get('pct_negative', 0)}%"),
                    _card("% Neutras", f"{overview.get('pct_neutral', 0)}%"),
                ],
            ),
            html.Div(
                className="app-card",
                style=CARD_STYLE,
                children=[
                    html.H4("Sentimento por categoria de aspeto"),
                    html.Div(category_tiles, style={"display": "flex", "flexWrap": "wrap", "gap": "16px"}),
                ],
            ),
        ]
    )
