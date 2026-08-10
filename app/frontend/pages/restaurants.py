"""
Restaurantes - ranking, destaque top/bottom 3, score vs volume, grau de
inspeção vs sentimento, comparação por distrito e drill-down por restaurante.
"""

import dash
import plotly.graph_objects as go
from dash import Input, Output, callback, dash_table, dcc, html
from dash.exceptions import PreventUpdate

from common import (
    ACCENT_COLOR,
    CARD_STYLE,
    CATEGORY_LABELS,
    CHART_LAYOUT,
    MUTED_COLOR,
    POLARITY_COLORS,
    TEXT_COLOR,
    fetch_restaurants,
    get_json,
    pivot_category_counts,
)
from filters import filter_inputs, filter_params, filter_sidebar, register_sidebar_toggle

dash.register_page(__name__, path="/restaurantes", name="Restaurantes", category="Dashboard", order=4)

CHART_HEIGHT = "420px"  # shared by every chart/table card on this page so rows line up


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
        "Clica numa linha da tabela ou num ponto do gráfico acima para veres o detalhe desse restaurante.",
        style={
            "color": MUTED_COLOR,
            "fontSize": "13px",
            "height": CHART_HEIGHT,
            "display": "flex",
            "alignItems": "center",
        },
    )


def _trend_figure(rows: list[dict], title: str) -> go.Figure:
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


def _top_bottom_card(title: str, rows: list[dict], accent: str) -> html.Div:
    if not rows:
        items = [html.Div("Sem dados suficientes.", style={"color": MUTED_COLOR, "fontSize": "13px"})]
    else:
        items = [
            html.Div(
                style={
                    "display": "flex",
                    "justifyContent": "space-between",
                    "padding": "8px 0",
                    "borderBottom": "1px solid #33343a" if i < len(rows) else "none",
                },
                children=[
                    html.Span(f"{i}. {r['name']}", style={"fontSize": "13px"}),
                    html.Span(f"{r['avg_crisp_score']:.2f}", style={"fontWeight": "bold", "color": accent}),
                ],
            )
            for i, r in enumerate(rows, start=1)
        ]

    return html.Div(
        className="app-card",
        style={**CARD_STYLE, "flex": "1", "minWidth": "280px", "padding": "16px"},
        children=[html.H4(title)] + items,
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
            dcc.Store(id="restaurant-drilldown-store"),
        ],
    )


@callback(Output("restaurants-content", "children"), *filter_inputs("restaurants"))
def render_restaurants(restaurant_ids, districts, categories, grades, polarities, aspect_categories, start_date, end_date):
    params = filter_params(restaurant_ids, districts, categories, grades, polarities, aspect_categories, start_date, end_date)

    rows = get_json("/analytics/restaurant-performance", [], params=params)
    plot_rows = [r for r in rows if r.get("avg_crisp_score") is not None]

    # ── Top 3 / Bottom 3 - always global, independent of the active filters ──
    unfiltered_rows = get_json("/analytics/restaurant-performance", [])
    unfiltered_plot_rows = [r for r in unfiltered_rows if r.get("avg_crisp_score") is not None]
    top3 = unfiltered_plot_rows[:3]
    bottom3 = list(reversed(unfiltered_plot_rows[-3:])) if unfiltered_plot_rows else []

    # ── Ranking table (rows carry "id" so click handlers get a stable id_restaurant) ──
    if not rows:
        table = html.Div("Ainda sem reviews associadas a restaurantes.", style={"color": "#999"})
    else:
        table_rows = [
            {
                "id": r["id_restaurant"],
                "name": r["name"],
                "district": r.get("district") or "—",
                "avg_crisp_score": round(r["avg_crisp_score"], 2) if r["avg_crisp_score"] is not None else None,
                "review_count": r["review_count"],
            }
            for r in rows
        ]
        table = dash_table.DataTable(
            id="restaurant-table",
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

    # ── Score vs volume (clickable: feeds the drill-down) ─────────
    scatter_fig = go.Figure(
        go.Scatter(
            x=[r["review_count"] for r in plot_rows],
            y=[r["avg_crisp_score"] for r in plot_rows],
            mode="markers+text",
            text=[r["name"] for r in plot_rows],
            textposition="top center",
            textfont=dict(color=TEXT_COLOR, size=10),
            customdata=[r["id_restaurant"] for r in plot_rows],
            marker=dict(size=14, color=[r["avg_crisp_score"] for r in plot_rows], colorscale="RdYlGn", cmin=0, cmax=1),
        )
    )
    scatter_fig.update_layout(
        title="Score fuzzy médio vs volume de reviews",
        xaxis_title="Nº de reviews",
        yaxis_title="Score fuzzy médio",
        **CHART_LAYOUT,
    )

    # ── Grau de inspeção vs sentimento ─────────────────────────────
    # One bar per grade: the average avg_crisp_score across every
    # restaurant that has that grade (a simple mean across restaurants,
    # not weighted by review_count).
    grade_scores: dict[str, list[float]] = {}
    for r in plot_rows:
        grade_scores.setdefault(r.get("inspection_grade") or "—", []).append(r["avg_crisp_score"])
    grades_sorted = sorted(grade_scores)
    grade_avgs = [sum(grade_scores[g]) / len(grade_scores[g]) for g in grades_sorted]

    grade_fig = go.Figure(
        go.Bar(
            x=grades_sorted,
            y=grade_avgs,
            marker=dict(color=grade_avgs, colorscale="RdYlGn", cmin=0, cmax=1),
        )
    )
    grade_fig.update_layout(
        title="Grau de inspeção vs sentimento",
        xaxis_title="Grau de inspeção",
        yaxis_title="Score fuzzy médio",
        **CHART_LAYOUT,
    )

    # ── Comparação por distrito (weighted by review_count) ─────────
    weighted: dict[str, tuple[float, int]] = {}
    for r in plot_rows:
        d = r.get("district") or "—"
        total, count = weighted.get(d, (0.0, 0))
        weighted[d] = (total + r["avg_crisp_score"] * r["review_count"], count + r["review_count"])
    districts_sorted = sorted(weighted, key=lambda d: weighted[d][0] / weighted[d][1], reverse=True)
    district_fig = go.Figure(
        go.Bar(
            x=districts_sorted,
            y=[weighted[d][0] / weighted[d][1] for d in districts_sorted],
            marker_color=ACCENT_COLOR,
        )
    )
    district_fig.update_layout(
        title="Score médio por distrito", xaxis_title="Distrito", yaxis_title="Score fuzzy médio", **CHART_LAYOUT
    )

    return html.Div(
        [
            _row(
                [
                    _top_bottom_card("Top 3 restaurantes", top3, POLARITY_COLORS["positive"]),
                    _top_bottom_card("Bottom 3 restaurantes", bottom3, POLARITY_COLORS["negative"]),
                ]
            ),
            _row(
                [
                    _chart_card(
                        html.Div(
                            [html.H4("Ranking de restaurantes"), table],
                            style={"height": CHART_HEIGHT, "overflowY": "auto"},
                        )
                    ),
                    _chart_card(
                        dcc.Graph(
                            id="restaurant-scatter-graph",
                            figure=scatter_fig,
                            config={"responsive": True},
                            style={"height": CHART_HEIGHT},
                        )
                    ),
                ]
            ),
            _row(
                [
                    _chart_card(
                        dcc.Graph(figure=grade_fig, config={"responsive": True}, style={"height": CHART_HEIGHT})
                    ),
                    _chart_card(
                        dcc.Graph(figure=district_fig, config={"responsive": True}, style={"height": CHART_HEIGHT})
                    ),
                ]
            ),
            _row(
                [
                    _chart_card(
                        [
                            html.Div(
                                style={
                                    "display": "flex",
                                    "justifyContent": "space-between",
                                    "alignItems": "center",
                                    "marginBottom": "8px",
                                },
                                children=[
                                    html.H4("Drill-down por restaurante", style={"margin": 0}),
                                    html.Button(
                                        "Limpar seleção",
                                        id="restaurant-drilldown-clear",
                                        n_clicks=0,
                                        className="clear-filters-btn",
                                        style={"width": "auto", "padding": "6px 12px", "fontSize": "12px"},
                                    ),
                                ],
                            ),
                            html.Div(id="restaurant-drilldown-content", children=_drilldown_placeholder()),
                        ]
                    ),
                ]
            ),
        ]
    )


@callback(
    Output("restaurant-drilldown-store", "data", allow_duplicate=True),
    Input("restaurant-table", "active_cell"),
    Input("restaurant-scatter-graph", "clickData"),
    prevent_initial_call=True,
)
def set_drilldown_restaurant(active_cell, click_data):
    """Clicking a table row or a scatter point selects that restaurant for the drill-down below."""
    triggered_id = dash.ctx.triggered_id
    if triggered_id == "restaurant-table":
        if not active_cell or active_cell.get("row_id") is None:
            raise PreventUpdate
        return active_cell["row_id"]
    if triggered_id == "restaurant-scatter-graph":
        if not click_data:
            raise PreventUpdate
        return click_data["points"][0]["customdata"]
    raise PreventUpdate


@callback(
    Output("restaurant-drilldown-store", "data", allow_duplicate=True),
    Output("restaurant-table", "active_cell", allow_duplicate=True),
    Input("restaurant-drilldown-clear", "n_clicks"),
    prevent_initial_call=True,
)
def clear_drilldown_restaurant(n_clicks):
    """Lets the user close the drill-down (and the table's highlighted cell) without refreshing the page."""
    if not n_clicks:
        raise PreventUpdate
    return None, None


@callback(
    Output("restaurant-drilldown-content", "children"),
    Input("restaurant-drilldown-store", "data"),
    *filter_inputs("restaurants"),
)
def render_restaurant_drilldown(
    restaurant_id, restaurant_ids, districts, categories, grades, polarities, aspect_categories, start_date, end_date
):
    if not restaurant_id:
        return _drilldown_placeholder()

    name = next((r["name"] for r in fetch_restaurants() if r["id_restaurant"] == restaurant_id), "—")

    params = filter_params(
        restaurant_ids, districts, categories, grades, polarities, aspect_categories, start_date, end_date
    )
    params["restaurant_id"] = restaurant_id  # narrows to just this restaurant, other active filters still apply

    category_rows = get_json("/analytics/sentiment-by-category", [], params=params)
    pivot = pivot_category_counts(category_rows)
    cats = [c for c in pivot if sum(pivot[c].values()) > 0]
    labels = [CATEGORY_LABELS.get(c, c) for c in cats]

    category_fig = go.Figure()
    for polarity, color in POLARITY_COLORS.items():
        category_fig.add_bar(
            name=polarity.capitalize(),
            x=labels,
            y=[pivot[c][polarity] for c in cats],
            marker_color=color,
        )
    category_fig.update_layout(barmode="stack", title=f'Categorias - "{name}"', **CHART_LAYOUT)

    trend_rows = get_json("/analytics/sentiment-over-time", [], params=params)
    trend_fig = _trend_figure(trend_rows, f'Tendência - "{name}"')

    return _row(
        [
            _chart_card(dcc.Graph(figure=category_fig, config={"responsive": True}, style={"height": CHART_HEIGHT})),
            _chart_card(dcc.Graph(figure=trend_fig, config={"responsive": True}, style={"height": CHART_HEIGHT})),
        ]
    )


register_sidebar_toggle("restaurants")
