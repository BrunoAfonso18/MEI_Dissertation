"""Categorias - sentimento por categoria (bar chart) e volume por categoria (radar)."""

import dash
import plotly.graph_objects as go
from dash import Input, Output, callback, dcc, html

from common import CARD_STYLE, CATEGORY_LABELS, CHART_LAYOUT, POLARITY_COLORS, fetch_restaurants, get_json, pivot_category_counts
from filters import filter_bar, filter_params

dash.register_page(__name__, path="/categorias", name="Categorias", category="Dashboard", order=2)


def _chart_card(children) -> html.Div:
    return html.Div(
        className="app-card",
        style={**CARD_STYLE, "flex": "1", "minWidth": "420px", "padding": "16px"},
        children=children,
    )


def layout():
    return html.Div(
        [
            html.H2("Categorias de Aspeto"),
            filter_bar("categories", fetch_restaurants()),
            html.Div(id="categories-content"),
        ]
    )


@callback(
    Output("categories-content", "children"),
    Input("categories-district", "value"),
    Input("categories-restaurant", "value"),
    Input("categories-date-range", "start_date"),
    Input("categories-date-range", "end_date"),
)
def render_categories(district, restaurant_id, start_date, end_date):
    params = filter_params(district, restaurant_id, start_date, end_date)

    rows = get_json("/analytics/sentiment-by-category", [], params=params)
    pivot = pivot_category_counts(rows)
    cats = list(pivot.keys())
    labels = [CATEGORY_LABELS.get(c, c) for c in cats]

    bar_fig = go.Figure()
    for polarity, color in POLARITY_COLORS.items():
        bar_fig.add_bar(
            name=polarity.capitalize(),
            x=labels,
            y=[pivot[c][polarity] for c in cats],
            marker_color=color,
        )
    bar_fig.update_layout(barmode="stack", title="Sentimento por categoria de aspeto", **CHART_LAYOUT)

    totals = [sum(pivot[c].values()) for c in cats]
    radar_fig = go.Figure(
        go.Scatterpolar(r=totals + totals[:1], theta=labels + labels[:1], fill="toself", name="Nº de menções")
    )
    radar_fig.update_layout(
        polar=dict(bgcolor=CARD_STYLE["backgroundColor"], radialaxis=dict(visible=True)),
        title="Volume de menções por categoria",
        showlegend=False,
        **CHART_LAYOUT,
    )

    return html.Div(
        style={"display": "flex", "gap": "20px", "flexWrap": "wrap"},
        children=[
            _chart_card(dcc.Graph(figure=bar_fig)),
            _chart_card(dcc.Graph(figure=radar_fig)),
        ],
    )
