"""
Categorias - sentimento por categoria (barra + radar), resumo tabular e
tendência por categoria (sparklines).
"""

import dash
import plotly.graph_objects as go
from dash import Output, callback, dash_table, dcc, html

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

dash.register_page(__name__, path="/categorias", name="Categorias", category="Dashboard", order=2)

CHART_HEIGHT = "420px"  # shared by every chart card on this page so rows line up


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


def _sparkline(values: list[float]) -> dcc.Graph:
    """A tiny, chrome-free line chart - static (no hover/zoom), sized to sit inside a tile."""
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


def layout():
    return html.Div(
        className="dashboard-layout",
        children=[
            html.Div(
                className="dashboard-main",
                children=[
                    html.H2("Categorias de Aspeto"),
                    dcc.Loading(html.Div(id="categories-content"), type="circle"),
                ],
            ),
            filter_sidebar("categories", fetch_restaurants()),
        ],
    )


@callback(Output("categories-content", "children"), *filter_inputs("categories"))
def render_categories(restaurant_ids, districts, categories, grades, polarities, aspect_categories, start_date, end_date):
    params = filter_params(restaurant_ids, districts, categories, grades, polarities, aspect_categories, start_date, end_date)

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

    # ── Resumo por categoria (tabela) + menções por review ──────────
    overview = get_json("/analytics/overview", {}, params=params)
    total_reviews = overview.get("total_reviews", 0)

    weighted_score = {cat: [0.0, 0] for cat in CATEGORY_LABELS}
    for r in rows:
        cat = r.get("category")
        avg = r.get("avg_crisp_score")
        count = r.get("count", 0)
        if cat not in weighted_score or avg is None:
            continue
        weighted_score[cat][0] += avg * count
        weighted_score[cat][1] += count

    table_rows = []
    for cat, label in CATEGORY_LABELS.items():
        counts = pivot[cat]
        total = sum(counts.values())
        total_score, total_count = weighted_score[cat]
        avg_score = total_score / total_count if total_count else None
        table_rows.append(
            {
                "id": cat,
                "category": label,
                "count": total,
                "per_review": round(total / total_reviews, 2) if total_reviews else 0,
                "pct_positive": round(100 * counts["positive"] / total, 1) if total else 0,
                "pct_negative": round(100 * counts["negative"] / total, 1) if total else 0,
                "pct_neutral": round(100 * counts["neutral"] / total, 1) if total else 0,
                "avg_score": round(avg_score, 2) if avg_score is not None else None,
            }
        )
    summary_table = dash_table.DataTable(
        id="category-summary-table",
        columns=[
            {"name": "Categoria", "id": "category"},
            {"name": "Nº menções", "id": "count"},
            {"name": "Menções/review", "id": "per_review"},
            {"name": "% Positivas", "id": "pct_positive"},
            {"name": "% Negativas", "id": "pct_negative"},
            {"name": "% Neutras", "id": "pct_neutral"},
            {"name": "Score médio", "id": "avg_score"},
        ],
        data=table_rows,
        style_header={"backgroundColor": "#111", "color": TEXT_COLOR, "fontWeight": "bold"},
        style_cell={"backgroundColor": CARD_STYLE["backgroundColor"], "color": TEXT_COLOR, "border": "1px solid #333", "padding": "6px"},
        style_as_list_view=True,
        sort_action="native",
        page_size=9,
    )

    # ── Tendência por categoria (sparklines) ─────────────────────────
    # aspect_categories already selected in the sidebar filter are respected
    # (an excluded category is shown with no data, without an extra
    # request), and categories with zero mentions skip the request too -
    # each category needs its own /sentiment-over-time call since the
    # endpoint doesn't break results down by category, so this keeps the
    # page from firing a request for tiles that would come back empty
    # anyway (only category-summary-table's own "count" is needed to know).
    category_tiles = []
    for cat, label in CATEGORY_LABELS.items():
        has_data = sum(pivot[cat].values()) > 0
        if not has_data or (aspect_categories and cat not in aspect_categories):
            values = []
        else:
            cat_params = dict(params)
            cat_params["aspect_category"] = cat
            trend_rows = get_json("/analytics/sentiment-over-time", [], params=cat_params)
            by_date: dict[str, int] = {}
            for r in trend_rows:
                by_date[r["date"]] = by_date.get(r["date"], 0) + r["count"]
            values = [by_date[d] for d in sorted(by_date)]

        tile_children = [
            html.Div(label, style={"fontSize": "12px", "fontWeight": "bold"}),
            html.Div(str(sum(values)), style={"fontSize": "16px", "fontWeight": "bold", "marginTop": "2px"}),
        ]
        if len(values) > 1:
            tile_children.append(_sparkline(values))
        else:
            tile_children.append(html.Div("Sem dados", style={"fontSize": "11px", "color": MUTED_COLOR, "marginTop": "8px"}))
        category_tiles.append(html.Div(tile_children, style={"width": "140px"}))

    return html.Div(
        [
            html.Div(
                className="app-card",
                style={**CARD_STYLE, "marginBottom": "20px"},
                children=[
                    html.H4("Tendência por categoria"),
                    html.Div(category_tiles, style={"display": "flex", "flexWrap": "wrap", "gap": "16px"}),
                ],
            ),
            _row(
                [
                    _chart_card(
                        dcc.Graph(figure=bar_fig, config={"responsive": True}, style={"height": CHART_HEIGHT})
                    ),
                    _chart_card(dcc.Graph(figure=radar_fig, config={"responsive": True}, style={"height": CHART_HEIGHT})),
                ]
            ),
            html.Div(
                className="app-card",
                style={**CARD_STYLE, "marginBottom": "20px"},
                children=[html.H4("Resumo por categoria"), summary_table],
            ),
        ]
    )


register_sidebar_toggle("categories")
