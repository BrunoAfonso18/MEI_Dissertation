"""
Collapsible filter sidebar for the Dashboard pages, rendered on the right of
each page's content.

Layout: [dashboard-main (grows/shrinks)] [toggle tab] [filter panel]. The
panel is grouped into three stacked sections - Restaurantes, Sentimento,
Período - starting with Restaurantes expanded. The toggle tab stays visible
even when the panel is collapsed (width -> 0) so it can always be reopened;
charts live in flex containers so they reflow automatically as the panel's
width changes, and dcc.Graph is set to responsive so Plotly redraws itself
to fit.

A filter change is the "event" that triggers a fresh /analytics/* request
built from whatever's selected (per the dashboard -> API -> DW -> render
flow); an empty selection on any field means "no filter on that field", so
with nothing selected the charts show the total.
"""

from dash import Input, Output, callback, dcc, html

from common import CATEGORY_LABELS, POLARITY_COLORS, TEXT_COLOR

MULTI_FIELDS = ["restaurant", "district", "category", "grade", "polarity", "aspect-category"]

_LABEL_STYLE = {"fontSize": "12px", "color": TEXT_COLOR, "marginBottom": "4px", "display": "block"}
_CONTROL_STYLE = {"color": "#000", "marginBottom": "12px"}


def _field(id_prefix: str, key: str, label: str, options: list[dict]) -> html.Div:
    return html.Div(
        [
            html.Label(label, style=_LABEL_STYLE),
            dcc.Dropdown(
                id=f"{id_prefix}-{key}",
                options=options,
                value=[],
                multi=True,
                placeholder="Todos",
                style=_CONTROL_STYLE,
            ),
        ]
    )


def _section(title: str, children: list, open_by_default: bool = False) -> html.Details:
    return html.Details(
        open=open_by_default,
        className="filter-section",
        children=[html.Summary(title, className="filter-section-title"), *children],
    )


def filter_sidebar(id_prefix: str, restaurants: list[dict]) -> html.Div:
    districts = sorted({r["district"] for r in restaurants if r.get("district")})
    categories = sorted({r["category"] for r in restaurants if r.get("category")})
    grades = sorted({r["inspection_grade"] for r in restaurants if r.get("inspection_grade")})

    return html.Div(
        className="filter-sidebar-wrapper",
        children=[
            html.Button("Filtros", id=f"{id_prefix}-sidebar-toggle", n_clicks=0, className="filter-sidebar-toggle"),
            html.Div(
                id=f"{id_prefix}-filter-sidebar",
                className="app-card filter-sidebar",
                children=[
                    _section(
                        "Restaurantes",
                        [
                            _field(id_prefix, "restaurant", "Restaurante", [
                                {"label": r["name"], "value": r["id_restaurant"]} for r in restaurants
                            ]),
                            _field(id_prefix, "district", "Distrito", [{"label": d, "value": d} for d in districts]),
                            _field(id_prefix, "category", "Categoria", [{"label": c, "value": c} for c in categories]),
                            _field(id_prefix, "grade", "Grau de inspeção", [{"label": g, "value": g} for g in grades]),
                        ],
                        open_by_default=True,
                    ),
                    _section(
                        "Sentimento",
                        [
                            _field(id_prefix, "polarity", "Polaridade", [
                                {"label": p.capitalize(), "value": p} for p in POLARITY_COLORS
                            ]),
                            _field(id_prefix, "aspect-category", "Categoria do aspeto", [
                                {"label": label, "value": code} for code, label in CATEGORY_LABELS.items()
                            ]),
                        ],
                    ),
                    _section(
                        "Período",
                        [
                            html.Div(
                                [
                                    html.Label("Intervalo de datas (opcional)", style=_LABEL_STYLE),
                                    dcc.DatePickerRange(
                                        id=f"{id_prefix}-date-range",
                                        display_format="YYYY-MM-DD",
                                        start_date_placeholder_text="Início",
                                        end_date_placeholder_text="Fim",
                                    ),
                                    html.Div(
                                        "Sem datas selecionadas, mostra o total.",
                                        style={"fontSize": "11px", "color": "#96979d", "marginTop": "6px"},
                                    ),
                                ]
                            ),
                        ],
                    ),
                ],
            ),
        ],
    )


def filter_inputs(id_prefix: str) -> list[Input]:
    """Ordered Input list matching filter_params()'s positional arguments."""
    inputs = [Input(f"{id_prefix}-{key}", "value") for key in MULTI_FIELDS]
    inputs += [Input(f"{id_prefix}-date-range", "start_date"), Input(f"{id_prefix}-date-range", "end_date")]
    return inputs


def filter_params(restaurant_ids, districts, categories, grades, polarities, aspect_categories, start_date, end_date) -> dict:
    """Builds /analytics/* query params from the filter sidebar's current values, omitting unset fields."""
    params = {}

    def add(key, value):
        if value:
            params[key] = value

    add("restaurant_id", restaurant_ids)
    add("district", districts)
    add("category", categories)
    add("inspection_grade", grades)
    add("sentiment_polarity", polarities)
    add("aspect_category", aspect_categories)
    if start_date:
        params["start_date"] = start_date
    if end_date:
        params["end_date"] = end_date
    return params


def register_sidebar_toggle(id_prefix: str) -> None:
    """Registers the collapse/expand callback for one page's filter sidebar."""

    @callback(
        Output(f"{id_prefix}-filter-sidebar", "className"),
        Output(f"{id_prefix}-sidebar-toggle", "children"),
        Input(f"{id_prefix}-sidebar-toggle", "n_clicks"),
    )
    def _toggle_sidebar(n_clicks):
        collapsed = (n_clicks or 0) % 2 == 1
        cls = "app-card filter-sidebar collapsed" if collapsed else "app-card filter-sidebar"
        label = "Filtros ▸" if collapsed else "Filtros ✕"
        return cls, label
