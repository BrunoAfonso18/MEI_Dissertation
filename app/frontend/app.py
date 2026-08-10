"""
Plotly Dash frontend - ABSA sentiment smiley + review submission + dashboard.

Uses Dash's Pages feature (the direct equivalent of Streamlit's
st.navigation) so the app keeps the same sidebar + 5-page structure that was
built while experimenting with Streamlit: "Submeter Review" under a Reviews
section, and "Visão Geral" / "Categorias" / "Aspetos & Tendência" /
"Restaurantes" under a Dashboard section. Only the currently active page's
layout is mounted, so background components (like the live smiley's
Interval) only run while that page is actually open.

Run with:
    python app/frontend/app.py
"""

import os

import dash
from dash import Dash, Input, Output, callback, dcc, html

from common import DARK_BG, FONT_FAMILY, TEXT_COLOR

app = Dash(__name__, use_pages=True, pages_folder="pages", suppress_callback_exceptions=True)
app.title = "ABSA Sentiment Dashboard"
server = app.server


def _nav_section(title: str, category: str):
    links = [
        dcc.Link(
            page["name"],
            href=page["path"],
            id={"type": "nav-link", "path": page["path"]},
            className="nav-link",
        )
        for page in dash.page_registry.values()
        if page.get("category") == category
    ]
    return html.Div(
        [
            html.Div(title, className="nav-section-title"),
            *links,
        ]
    )


sidebar = html.Div(
    className="sidebar",
    children=[
        html.Div("ABSA Sentiment", className="sidebar-brand"),
        html.Div("Analytics Dashboard", className="sidebar-subtitle"),
        _nav_section("Reviews", "Reviews"),
        _nav_section("Dashboard", "Dashboard"),
    ],
)

app.layout = html.Div(
    style={
        "backgroundColor": DARK_BG,
        "color": TEXT_COLOR,
        "minHeight": "100vh",
        "fontFamily": FONT_FAMILY,
        "display": "flex",
    },
    children=[
        dcc.Location(id="url"),
        sidebar,
        html.Div(
            dash.page_container,
            style={"flex": "1", "padding": "32px", "minWidth": 0},
        ),
    ],
)


@callback(
    Output({"type": "nav-link", "path": dash.ALL}, "className"),
    Input("url", "pathname"),
)
def _highlight_active_link(pathname):
    matched_paths = [item["id"]["path"] for item in dash.ctx.outputs_list]
    return [
        "nav-link nav-link-active" if path == pathname else "nav-link"
        for path in matched_paths
    ]


def main() -> None:
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "8050")), debug=False)


if __name__ == "__main__":
    main()
