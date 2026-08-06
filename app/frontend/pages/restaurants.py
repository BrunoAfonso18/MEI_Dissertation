"""Restaurantes - ranking (tabela) e score fuzzy médio vs volume de reviews (scatter)."""

import dash
import plotly.graph_objects as go
from dash import Output, callback, dash_table, dcc, html

from common import CARD_STYLE, CHART_LAYOUT, TEXT_COLOR, fetch_restaurants, get_json
from filters import filter_inputs, filter_params, filter_sidebar, register_sidebar_toggle

dash.register_page(__name__, path="/restaurantes", name="Restaurantes", category="Dashboard", order=4)


def _chart_card(children) -> html.Div:
    return html.Div(
        className="app-card",
        style={**CARD_STYLE, "flex": "1", "minWidth": "420px", "padding": "16px"},
        children=children,
    )


def _ranking_table(rows: list[dict]):
    if not rows:
        return html.Div("Ainda sem reviews associadas a restaurantes.", style={"color": "#999"})

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
        style_cell={"backgroundColor": CARD_STYLE["backgroundColor"], "color": TEXT_COLOR, "border": "1px solid #333", "padding": "6px"},
        style_as_list_view=True,
        sort_action="native",
        page_size=8,
    )


def layout():
    return html.Div(
        className="dashboard-layout",
        children=[
            html.Div(
                className="dashboard-main",
                children=[
                    html.H2("Restaurantes"),
                    html.Div(id="restaurants-content"),
                ],
            ),
            filter_sidebar("restaurants", fetch_restaurants()),
        ],
    )


@callback(Output("restaurants-content", "children"), *filter_inputs("restaurants"))
def render_restaurants(restaurant_ids, districts, categories, grades, polarities, aspect_categories, start_date, end_date):
    params = filter_params(restaurant_ids, districts, categories, grades, polarities, aspect_categories, start_date, end_date)

    rows = get_json("/analytics/restaurant-performance", [], params=params)
    plot_rows = [r for r in rows if r.get("avg_crisp_score") is not None]

    scatter_fig = go.Figure(
        go.Scatter(
            x=[r["review_count"] for r in plot_rows],
            y=[r["avg_crisp_score"] for r in plot_rows],
            mode="markers+text",
            text=[r["name"] for r in plot_rows],
            textposition="top center",
            textfont=dict(color=TEXT_COLOR, size=10),
            marker=dict(size=14, color=[r["avg_crisp_score"] for r in plot_rows], colorscale="RdYlGn", cmin=0, cmax=1),
        )
    )
    scatter_fig.update_layout(
        title="Score fuzzy médio vs volume de reviews",
        xaxis_title="Nº de reviews",
        yaxis_title="Score fuzzy médio",
        **CHART_LAYOUT,
    )

    return html.Div(
        style={"display": "flex", "gap": "20px", "flexWrap": "wrap"},
        children=[
            _chart_card([html.H4("Ranking de restaurantes"), _ranking_table(rows)]),
            _chart_card(dcc.Graph(figure=scatter_fig, config={"responsive": True})),
        ],
    )


register_sidebar_toggle("restaurants")
