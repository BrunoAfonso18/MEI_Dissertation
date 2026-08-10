"""
Aspetos & Tendência - leaderboards de aspetos (criticados/elogiados),
evolução temporal, drill-down por aspeto e sazonalidade.
"""

import dash
import plotly.graph_objects as go
from dash import Input, Output, callback, dcc, html
from dash.exceptions import PreventUpdate

from common import CARD_STYLE, CHART_LAYOUT, MUTED_COLOR, POLARITY_COLORS, fetch_restaurants, get_json
from filters import filter_inputs, filter_params, filter_sidebar, register_sidebar_toggle

dash.register_page(__name__, path="/aspetos-tendencia", name="Aspetos & Tendência", category="Dashboard", order=3)

DOW_LABELS = ["Dom", "Seg", "Ter", "Qua", "Qui", "Sex", "Sáb"]  # index = Postgres EXTRACT(DOW)
MONTH_LABELS = ["Jan", "Fev", "Mar", "Abr", "Mai", "Jun", "Jul", "Ago", "Set", "Out", "Nov", "Dez"]


def _chart_card(children) -> html.Div:
    return html.Div(
        className="app-card",
        style={**CARD_STYLE, "flex": "1", "minWidth": "420px", "padding": "16px"},
        children=children,
    )


def _row(children) -> html.Div:
    return html.Div(
        style={"display": "flex", "gap": "20px", "flexWrap": "wrap", "marginBottom": "20px"},
        children=children,
    )


def _drilldown_placeholder() -> html.Div:
    return html.Div(
        "Clica numa barra num dos gráficos acima para veres a evolução temporal desse aspeto.",
        style={
            "color": MUTED_COLOR,
            "fontSize": "13px",
            "height": "420px",
            "display": "flex",
            "alignItems": "center",
        },
    )


def _trend_figure(rows: list[dict], title: str) -> go.Figure:
    """Shared by the overall trend chart and the per-aspect drill-down chart."""
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
    fig.update_layout(title=title, xaxis_title="Data", yaxis_title="Nº de menções", **CHART_LAYOUT)
    return fig


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
            dcc.Store(id="drilldown-aspect-store"),
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

    positive_rows = sorted(
        get_json("/analytics/top-positive-aspects", [], params=params), key=lambda r: r["count"]
    )
    positive_fig = go.Figure(
        go.Bar(
            x=[r["count"] for r in positive_rows],
            y=[r["aspect"] for r in positive_rows],
            orientation="h",
            marker_color=POLARITY_COLORS["positive"],
        )
    )
    positive_fig.update_layout(title="Aspetos mais elogiados", xaxis_title="Nº de menções positivas", **CHART_LAYOUT)

    trend_rows = get_json("/analytics/sentiment-over-time", [], params=params)
    trend_fig = _trend_figure(trend_rows, "Sentimento ao longo do tempo")

    seasonality_rows = get_json("/analytics/seasonality", [], params=params)
    z = [[0] * 7 for _ in range(12)]
    for r in seasonality_rows:
        dow, month = r.get("day_of_week"), r.get("month")
        if dow is not None and month is not None and 0 <= dow <= 6 and 1 <= month <= 12:
            z[month - 1][dow] = r["count"]
    seasonality_fig = go.Figure(
        go.Heatmap(
            z=z,
            x=DOW_LABELS,
            y=MONTH_LABELS,
            colorscale="Blues",
            hovertemplate="%{y}, %{x}: %{z} menções<extra></extra>",
        )
    )
    seasonality_fig.update_layout(title="Sazonalidade - menções por dia da semana e mês", **CHART_LAYOUT)

    return html.Div(
        [
            _row(
                [
                    _chart_card(dcc.Graph(id="top-negative-graph", figure=negative_fig, config={"responsive": True})),
                    _chart_card(dcc.Graph(id="top-positive-graph", figure=positive_fig, config={"responsive": True})),
                ]
            ),
            _row(
                [
                    _chart_card(
                        dcc.Graph(figure=trend_fig, config={"responsive": True}, style={"height": "420px"})
                    ),
                    _chart_card(
                        [
                            html.H4("Drill-down por aspeto"),
                            html.Div(id="aspect-drilldown-content", children=_drilldown_placeholder()),
                        ]
                    ),
                ]
            ),
            _row([_chart_card(dcc.Graph(figure=seasonality_fig, config={"responsive": True}))]),
        ]
    )


@callback(
    Output("drilldown-aspect-store", "data"),
    Input("top-negative-graph", "clickData"),
    Input("top-positive-graph", "clickData"),
    prevent_initial_call=True,
)
def set_drilldown_aspect(_negative_click, _positive_click):
    """Clicking a bar in either leaderboard selects that aspect for the drill-down chart below."""
    triggered_id = dash.ctx.triggered_id
    click = _negative_click if triggered_id == "top-negative-graph" else _positive_click
    if not click:
        raise PreventUpdate
    return click["points"][0]["y"]


@callback(
    Output("aspect-drilldown-content", "children"),
    Input("drilldown-aspect-store", "data"),
    *filter_inputs("aspects-trend"),
)
def render_drilldown(
    aspect_term, restaurant_ids, districts, categories, grades, polarities, aspect_categories, start_date, end_date
):
    if not aspect_term:
        return _drilldown_placeholder()

    params = filter_params(
        restaurant_ids, districts, categories, grades, polarities, aspect_categories, start_date, end_date
    )
    params["aspect_term"] = aspect_term

    rows = get_json("/analytics/sentiment-over-time", [], params=params)
    fig = _trend_figure(rows, f'Tendência de "{aspect_term}"')
    return dcc.Graph(figure=fig, config={"responsive": True}, style={"height": "420px"})


register_sidebar_toggle("aspects-trend")
