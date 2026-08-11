"""
Visão Geral - KPI cards com sparkline, cards de destaque (melhor/pior
restaurante e distrito), smileys por categoria, donuts de distribuição e
polaridade, e sazonalidade das reviews.
"""

import dash
import plotly.graph_objects as go
from dash import Output, callback, dcc, html

from common import (
    ACCENT_COLOR,
    CARD_STYLE,
    CATEGORY_LABELS,
    CHART_LAYOUT,
    MUTED_COLOR,
    POLARITY_COLORS,
    crisp_positivity,
    fetch_restaurants,
    frame_to_data_uri,
    get_json,
)
from filters import filter_inputs, filter_params, filter_sidebar, register_sidebar_toggle
from smiley import render_smiley

dash.register_page(__name__, path="/visao-geral", name="Visão Geral", category="Dashboard", order=1)

KPI_SMILEY_SIZE = (60, 60)
CATEGORY_SMILEY_SIZE = (90, 90)
CHART_HEIGHT = "420px"  # shared by every chart card on this page so rows line up
DOW_LABELS = ["Dom", "Seg", "Ter", "Qua", "Qui", "Sex", "Sáb"]  # index = Postgres EXTRACT(DOW)
MONTH_LABELS = ["Jan", "Fev", "Mar", "Abr", "Mai", "Jun", "Jul", "Ago", "Set", "Out", "Nov", "Dez"]


def _row(children) -> html.Div:
    return html.Div(
        style={"display": "flex", "gap": "20px", "flexWrap": "wrap", "marginBottom": "20px"},
        children=children,
    )


def _chart_card(children) -> html.Div:
    return html.Div(
        className="app-card",
        style={**CARD_STYLE, "flex": "1", "minWidth": "420px", "padding": "16px"},
        children=children,
    )


def _sparkline(values: list[float]) -> dcc.Graph:
    """A tiny, chrome-free line chart - static (no hover/zoom), sized to sit inside a KPI card."""
    fig = go.Figure(
        go.Scatter(
            y=values,
            mode="lines",
            line=dict(color=ACCENT_COLOR, width=2),
            fill="tozeroy",
            fillcolor="rgba(76, 124, 240, 0.15)",
        )
    )
    fig.update_layout(
        margin=dict(l=0, r=0, t=4, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        showlegend=False,
        height=36,
    )
    return dcc.Graph(
        figure=fig,
        config={"staticPlot": True},
        style={"height": "36px", "width": "100%", "marginTop": "6px"},
    )


def _card(label: str, value, sparkline_values: list[float] = None) -> html.Div:
    children = [
        html.Div(label, style={"fontSize": "12px", "color": MUTED_COLOR}),
        html.Div(str(value), style={"fontSize": "22px", "fontWeight": "bold"}),
    ]
    if sparkline_values and len(sparkline_values) > 1:
        children.append(_sparkline(sparkline_values))
    return html.Div(
        className="app-card",
        style={**CARD_STYLE, "padding": "16px 20px", "minWidth": "160px", "flex": "1"},
        children=children,
    )


def _highlight_card(label: str, name: str, score, accent: str) -> html.Div:
    return html.Div(
        className="app-card",
        style={**CARD_STYLE, "flex": "1", "minWidth": "200px", "padding": "16px 20px"},
        children=[
            html.Div(label, style={"fontSize": "12px", "color": MUTED_COLOR}),
            html.Div(name or "—", style={"fontSize": "16px", "fontWeight": "bold", "marginTop": "4px"}),
            html.Div(
                f"{score:.2f}" if score is not None else "Sem dados",
                style={"fontSize": "13px", "color": accent, "marginTop": "2px", "fontWeight": "bold"},
            ),
        ],
    )


def layout():
    return html.Div(
        className="dashboard-layout",
        children=[
            html.Div(
                className="dashboard-main",
                children=[
                    html.H2("Visão Geral"),
                    html.Div(id="overview-content"),
                ],
            ),
            filter_sidebar("overview", fetch_restaurants()),
        ],
    )


@callback(Output("overview-content", "children"), *filter_inputs("overview"))
def render_overview(restaurant_ids, districts, categories, grades, polarities, aspect_categories, start_date, end_date):
    params = filter_params(restaurant_ids, districts, categories, grades, polarities, aspect_categories, start_date, end_date)

    overview = get_json("/analytics/overview", {}, params=params)
    avg_score = overview.get("avg_crisp_score")

    category_rows = get_json("/analytics/sentiment-by-category", [], params=params)
    restaurant_rows = get_json("/analytics/restaurant-performance", [], params=params)
    trend_rows = get_json("/analytics/sentiment-over-time", [], params=params)
    reviews_trend_rows = get_json("/analytics/reviews-over-time", [], params=params)
    seasonality_rows = get_json("/analytics/seasonality", [], params=params)

    # ── Sparklines: one time series per KPI, derived from the same trend rows ──
    by_date: dict[str, dict[str, tuple[int, float]]] = {}
    for r in trend_rows:
        pol = r.get("polarity") or "neutral"
        by_date.setdefault(r["date"], {})[pol] = (r["count"], r.get("avg_crisp_score"))
    dates_sorted = sorted(by_date)

    score_sparkline, pct_pos_sparkline, pct_neg_sparkline, pct_neu_sparkline = [], [], [], []
    for d in dates_sorted:
        entries = by_date[d]
        total = sum(c for c, _ in entries.values())
        if not total:
            continue
        weighted_sum = sum(c * s for c, s in entries.values() if s is not None)
        score_sparkline.append(weighted_sum / total)
        pct_pos_sparkline.append(100 * entries.get("positive", (0, None))[0] / total)
        pct_neg_sparkline.append(100 * entries.get("negative", (0, None))[0] / total)
        pct_neu_sparkline.append(100 * entries.get("neutral", (0, None))[0] / total)

    reviews_sparkline = [r["count"] for r in sorted(reviews_trend_rows, key=lambda r: r["date"])]

    # ── Cards de destaque: melhor/pior restaurante e distrito ──────
    plot_restaurants = [r for r in restaurant_rows if r.get("avg_crisp_score") is not None]
    best_restaurant = plot_restaurants[0] if plot_restaurants else None
    worst_restaurant = plot_restaurants[-1] if plot_restaurants else None

    district_weighted: dict[str, tuple[float, int]] = {}
    for r in plot_restaurants:
        d = r.get("district") or "—"
        total, count = district_weighted.get(d, (0.0, 0))
        district_weighted[d] = (total + r["avg_crisp_score"] * r["review_count"], count + r["review_count"])
    district_avgs = {d: total / count for d, (total, count) in district_weighted.items() if count}
    best_district = max(district_avgs, key=district_avgs.get) if district_avgs else None
    worst_district = min(district_avgs, key=district_avgs.get) if district_avgs else None

    # ── Categorias (smileys, já existente) ──────────────────────────
    weighted = {cat: [0.0, 0] for cat in CATEGORY_LABELS}
    for r in category_rows:
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

    # ── Distribuição de reviews por distrito (donut) ────────────────
    district_counts: dict[str, int] = {}
    for r in plot_restaurants:
        d = r.get("district") or "—"
        district_counts[d] = district_counts.get(d, 0) + r["review_count"]
    district_donut = go.Figure(
        go.Pie(labels=list(district_counts.keys()), values=list(district_counts.values()), hole=0.55)
    )
    district_donut.update_layout(title="Distribuição de reviews por distrito", **CHART_LAYOUT)

    # ── Polaridade geral (donut) ─────────────────────────────────────
    polarity_counts = {"positive": 0, "neutral": 0, "negative": 0}
    for r in category_rows:
        pol = r.get("polarity")
        if pol in polarity_counts:
            polarity_counts[pol] += r.get("count", 0)
    polarity_donut = go.Figure(
        go.Pie(
            labels=[p.capitalize() for p in polarity_counts],
            values=list(polarity_counts.values()),
            hole=0.55,
            marker=dict(colors=[POLARITY_COLORS[p] for p in polarity_counts]),
        )
    )
    polarity_donut.update_layout(title="Polaridade geral", **CHART_LAYOUT)

    # ── Sazonalidade das reviews (heatmap dia da semana x mês) ──────
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
    seasonality_fig.update_layout(title="Sazonalidade das reviews", **CHART_LAYOUT)

    return html.Div(
        [
            _row(
                [
                    html.Div(
                        className="app-card",
                        style={
                            **CARD_STYLE,
                            "padding": "10px 20px",
                            "display": "flex",
                            "alignItems": "center",
                            "gap": "10px",
                            "flex": "1",
                            "minWidth": "160px",
                        },
                        children=[
                            html.Img(
                                src=frame_to_data_uri(render_smiley(crisp_positivity(avg_score), KPI_SMILEY_SIZE)),
                                style={"width": f"{KPI_SMILEY_SIZE[0]}px", "height": f"{KPI_SMILEY_SIZE[1]}px"},
                            ),
                            html.Div(
                                style={"flex": "1", "minWidth": 0},
                                children=[
                                    html.Div("Sentimento geral", style={"fontSize": "12px", "color": MUTED_COLOR}),
                                    html.Div(
                                        f"{avg_score:.2f}" if avg_score is not None else "—",
                                        style={"fontSize": "20px", "fontWeight": "bold"},
                                    ),
                                    _sparkline(score_sparkline) if len(score_sparkline) > 1 else None,
                                ],
                            ),
                        ],
                    ),
                    _card("Total de reviews", overview.get("total_reviews", 0), reviews_sparkline),
                    _card("% Positivas", f"{overview.get('pct_positive', 0)}%", pct_pos_sparkline),
                    _card("% Negativas", f"{overview.get('pct_negative', 0)}%", pct_neg_sparkline),
                    _card("% Neutras", f"{overview.get('pct_neutral', 0)}%", pct_neu_sparkline),
                ]
            ),
            _row(
                [
                    _highlight_card(
                        "Melhor restaurante",
                        best_restaurant["name"] if best_restaurant else None,
                        best_restaurant["avg_crisp_score"] if best_restaurant else None,
                        POLARITY_COLORS["positive"],
                    ),
                    _highlight_card(
                        "Restaurante a precisar de atenção",
                        worst_restaurant["name"] if worst_restaurant else None,
                        worst_restaurant["avg_crisp_score"] if worst_restaurant else None,
                        POLARITY_COLORS["negative"],
                    ),
                    _highlight_card(
                        "Melhor distrito", best_district, district_avgs.get(best_district), POLARITY_COLORS["positive"]
                    ),
                    _highlight_card(
                        "Distrito a precisar de atenção",
                        worst_district,
                        district_avgs.get(worst_district),
                        POLARITY_COLORS["negative"],
                    ),
                ]
            ),
            html.Div(
                className="app-card",
                style={**CARD_STYLE, "marginBottom": "20px"},
                children=[
                    html.H4("Sentimento por categoria de aspeto"),
                    html.Div(category_tiles, style={"display": "flex", "flexWrap": "wrap", "gap": "16px"}),
                ],
            ),
            _row(
                [
                    _chart_card(dcc.Graph(figure=district_donut, config={"responsive": True}, style={"height": CHART_HEIGHT})),
                    _chart_card(dcc.Graph(figure=polarity_donut, config={"responsive": True}, style={"height": CHART_HEIGHT})),
                ]
            ),
            _row([_chart_card(dcc.Graph(figure=seasonality_fig, config={"responsive": True}, style={"height": CHART_HEIGHT}))]),
        ]
    )


register_sidebar_toggle("overview")
