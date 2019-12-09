import dash
import sys
from dash.dependencies import Input, Output, State
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
import dash_table as tbl
import dash_daq as daq
from typing import Type

from negmas import NamedObject
from negmas.visualizers import *

# 'https://codepen.io/chriddyp/pen/bWLwgP.css',
external_style_sheets = [dbc.themes.CERULEAN]


def cli(debug=True):
    """Main entry point"""

    navbar = dbc.NavbarSimple(
        children=[
            dbc.NavItem(dbc.NavLink("Settings", href="#")),
            # dbc.DropdownMenu(
            #     nav=True,
            #     in_navbar=True,
            #     label="Menu",
            #     children=[
            #         dbc.DropdownMenuItem("Entry 1"),
            #         dbc.DropdownMenuItem("Entry 2"),
            #         dbc.DropdownMenuItem(divider=True),
            #         dbc.DropdownMenuItem("Entry 3"),
            #     ],
            # ),
        ],
        brand="NegMAS GUI",
        brand_href="#",
        sticky="top",
    )

    run_online = dbc.Card(
        dbc.CardBody(
            [
                html.P("Run a new component", className="card-text"),
                dbc.Select(
                    options=[
                        {"label": "SCMLWorld", "value": "negmas.apps.scml.SCMLWorld"},
                        {"label": "World", "value": "negmas.situated.World"},
                        {
                            "label": "Negotiation",
                            "value": "negmas.mechanisms.Mechanism",
                        },
                    ],
                    value="negmas.apps.scml.SCMLWorld",
                    id="new-type",
                    className="mt-1",
                ),
                dbc.Input(
                    placeholder="Config file path ...",
                    type="file",
                    value="",
                    id="new-config-path",
                    className="mt-1",
                ),
                dbc.Button("Run"),
            ]
        ),
        className="mt-3",
    )

    run_offline = dbc.Card(
        dbc.CardBody(
            [
                html.P("Monitor a component", className="card-text"),
                dbc.Input(
                    placeholder="Checkpoint folder ...",
                    type="file",
                    value="",
                    id="checkpoint-folder",
                    className="mt-1",
                ),
                dbc.Input(
                    placeholder="[Optional] component ID",
                    type="text",
                    value="",
                    id="checkpoint-id",
                    className="mt-1",
                ),
                dbc.Checklist(
                    options=[{"label": "Watch Folder", "value": "watch"}],
                    value=[],
                    id="checkpoint-options",
                    className="mt-1",
                ),
                dbc.Button("Monitor"),
            ]
        ),
        className="mt-3",
    )

    body = dbc.Container(
        [
            dbc.Row(
                [
                    dbc.Col(
                        [
                            dbc.Container(
                                [
                                    dbc.Row([html.H2("Load")]),
                                    dbc.Row(
                                        [
                                            dbc.Tabs(
                                                [
                                                    dbc.Tab(
                                                        run_online,
                                                        label="Tab 1",
                                                        tab_id="online",
                                                    ),
                                                    dbc.Tab(
                                                        run_offline,
                                                        label="Tab 2",
                                                        tab_id="offline",
                                                    ),
                                                ],
                                                id="open-tabs",
                                                active_tab="online",
                                            )
                                        ]
                                    ),
                                    dbc.Row([html.H2("Children")]),
                                    dbc.Row([html.P("Children will appear here")]),
                                ],
                                className="mt-0",
                            )
                        ],
                        md=3,
                    ),
                    dbc.Col(
                        [
                            html.H2("Showing xyz (ID fdfdsf)", id="basic"),
                            html.Div(id="main_widget"),
                        ],
                        md=9,
                    ),
                ]
            )
        ],
        className="mt-0",
    )
    app = dash.Dash(__name__, external_stylesheets=external_style_sheets)
    app.layout = html.Div([navbar, body], style={"width": "100%"})
    app.run_server(debug=debug)


if __name__ == "__main__":
    debug = False
    if len(sys.argv) > 1 and sys.argv[1] in ("debug", "--debug", "-d"):
        debug = True
    cli(debug=debug)
