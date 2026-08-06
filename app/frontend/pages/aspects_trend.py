"""Aspetos & Tendência - leaderboard de aspetos mais criticados + evolução temporal."""

import dash
import plotly.graph_objects as go
from dash import Output, callback, dcc, html

from common import CARD_STYLE, CHART_LAYOUT, POLARITY_COLORS, fetch_restaurants, get_json
from filters import filter_inputs, filter_params, filter_sidebar, register_sidebar_toggle

dash.register_page(__name__, path="/aspetos-tendencia", name="Aspetos & Tendência", category="Dashboard", order=3)


def _chart_card(children) -> html.Div:
    return html.Div(
        className="app-card",
        style={**CARD_STYLE, "flex": "1", "minWidth": "420px", "padding": "16px"},
        children=children,
    )


def layout():
    return html.Div(
        className="dashboard-layout",
        children=[
            html.Div(
                className="dashboard-main",
                children=[
                    html.H2("Aspetos & Tendência"),
                    html.Div(id="aspects-trend-content"),
                ],
            ),
            filter_sidebar("aspects-trend", fetch_restaurants()),
        ],
    )


@callback(Output("aspects-trend-content", "children"), *filter_inputs("aspects-trend"))
def render_aspects_trend(restaurant_ids, districts, categories, grades, polarities, aspect_categories, start_date, end_date):
    params = filter_params(restaurant_ids, districts, categories, grades, polarities, aspect_categories, start_date, end_date)

    negative_rows = sorted(
        get_json("/analytics/top-negative-aspects", [], params=params), key=lambda r: r["count"]
    )
    negative_fig = go.Figure(
        go.Bar(
            x=[r["count"] for r in negative_rows],
            y=[r["aspect"] for r in negative_rows],
            orientation="h",
            marker_color=POLARITY_COLORS["negative"],
        )
    )
    negative_fig.update_layout(title="Aspetos mais criticados", xaxis_title="Nº de menções negativas", **CHART_LAYOUT)

    trend_rows = get_json("/analytics/sentiment-over-time", [], params=params)
    by_polarity: dict[str, list[tuple[str, int]]] = {}
    for r in trend_rows:
        pol = r.get("polarity") or "neutral"
        by_polarity.setdefault(pol, []).append((r["date"], r["count"]))

    trend_fig = go.Figure()
    for polarity, color in POLARITY_COLORS.items():
        points = sorted(by_polarity.get(polarity, []))
        trend_fig.add_trace(
            go.Scatter(
                x=[p[0] for p in points],
                y=[p[1] for p in points],
                mode="lines+markers",
                name=polarity.capitalize(),
                line_color=color,
            )
        )
    trend_fig.update_layout(title="Sentimento ao longo do tempo", xaxis_title="Data", yaxis_title="Nº de menções", **CHART_LAYOUT)

    return html.Div(
        style={"display": "flex", "gap": "20px", "flexWrap": "wrap"},
        children=[
            _chart_card(dcc.Graph(figure=negative_fig, config={"responsive": True})),
            _chart_card(dcc.Graph(figure=trend_fig, config={"responsive": True})),
        ],
    )


register_sidebar_toggle("aspects-trend")
