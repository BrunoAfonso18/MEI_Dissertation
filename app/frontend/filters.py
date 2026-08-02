"""
Shared filter bar (district, restaurant, date range) for the Dashboard pages.

Every Dashboard page renders this bar plus an empty content container in its
layout(); a callback listens for filter changes and re-fetches from the
/analytics/* endpoints with the active filters as query params, so each
filter change is the "event" that triggers a fresh SQL query against the
warehouse (per the dashboard -> API -> DW -> render flow).
"""

from dash import dcc, html

from common import CARD_STYLE, TEXT_COLOR

ALL_VALUE = "all"

_LABEL_STYLE = {"fontSize": "12px", "color": TEXT_COLOR, "marginBottom": "4px", "display": "block"}
_CONTROL_STYLE = {"minWidth": "200px", "color": "#000"}


def filter_bar(id_prefix: str, restaurants: list[dict]) -> html.Div:
    districts = sorted({r["district"] for r in restaurants if r.get("district")})
    district_options = [{"label": "Todos os distritos", "value": ALL_VALUE}] + [
        {"label": d, "value": d} for d in districts
    ]
    restaurant_options = [{"label": "Todos os restaurantes", "value": ALL_VALUE}] + [
        {"label": r["name"], "value": r["id_restaurant"]} for r in restaurants
    ]

    return html.Div(
        className="app-card",
        style={
            **CARD_STYLE,
            "display": "flex",
            "gap": "20px",
            "flexWrap": "wrap",
            "alignItems": "flex-end",
            "marginBottom": "20px",
            "padding": "16px 20px",
        },
        children=[
            html.Div(
                [
                    html.Label("Distrito", style=_LABEL_STYLE),
                    dcc.Dropdown(
                        id=f"{id_prefix}-district",
                        options=district_options,
                        value=ALL_VALUE,
                        clearable=False,
                        style=_CONTROL_STYLE,
                    ),
                ]
            ),
            html.Div(
                [
                    html.Label("Restaurante", style=_LABEL_STYLE),
                    dcc.Dropdown(
                        id=f"{id_prefix}-restaurant",
                        options=restaurant_options,
                        value=ALL_VALUE,
                        clearable=False,
                        style=_CONTROL_STYLE,
                    ),
                ]
            ),
            html.Div(
                [
                    html.Label("Intervalo de datas", style=_LABEL_STYLE),
                    dcc.DatePickerRange(
                        id=f"{id_prefix}-date-range",
                        display_format="YYYY-MM-DD",
                        start_date_placeholder_text="Início",
                        end_date_placeholder_text="Fim",
                    ),
                ]
            ),
        ],
    )


def filter_params(district, restaurant_id, start_date, end_date) -> dict:
    """Builds /analytics/* query params from the filter bar's current values, omitting unset ones."""
    params = {}
    if district and district != ALL_VALUE:
        params["district"] = district
    if restaurant_id and restaurant_id != ALL_VALUE:
        params["restaurant_id"] = restaurant_id
    if start_date:
        params["start_date"] = start_date
    if end_date:
        params["end_date"] = end_date
    return params
