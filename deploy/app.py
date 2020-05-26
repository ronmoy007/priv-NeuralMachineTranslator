'''
TODOS:
1. Con más cuidades, solo cargar fza de ventas de esa cuidad
'''
from src.db_utils import get_data
from src.visualize import ShortestPath

from dash.exceptions import PreventUpdate
from dash.dependencies import Input, Output, State
from datetime import datetime as dt
from datetime import date, timedelta
from textwrap import dedent
from folium import plugins
from folium.plugins import HeatMap
from folium.plugins import MarkerCluster


import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
import dash_table
import pandas as pd
import numpy as np
import folium
from flask import Flask, send_from_directory
import os


UPLOAD_DIRECTORY = "files"

if not os.path.exists(UPLOAD_DIRECTORY):
    os.makedirs(UPLOAD_DIRECTORY)

#---------------- INICIALIZE ----------------
print(dcc.__version__) # 0.6.0 or above is required
external_stylesheets = [dbc.themes.BOOTSTRAP]
server = Flask(__name__)
app = dash.Dash(server=server,  meta_tags=[{"name": "viewport", "content": "width=device-width"}])
#server = app.server
app.title= "Problema del vendedor viajero"

app.config.suppress_callback_exceptions = True


# ============================ ELEMENTS ============================
def markdown_popup():
    return html.Div(
        id="markdown",
        className="modal",
        style={"display": "none"},
        children=(
            html.Div(
                className="markdown-container",
                children=[
                    html.Div(
                        className="close-container",
                        children=html.Button(
                            "Cerrar",
                            id="markdown_close",
                            n_clicks=0,
                            className="closeButton",
                        ),
                    ),
                    html.Div(
                        className="markdown-text",
                        children=[
                            dcc.Markdown(
                                children=dedent(
                                    """
                                # Proyecto Final

                                ### Contexto del problema

                                En una microfinanciera, los colaboradores de la fuerza de ventas realizan reuniones de seguimiento con sus clientes para asegurarse de que el monto total de la cuota del grupo sea cubierta.

                                El principal problema que presentan los colaboradores es que las visitas no tienen ningún orden asignado, traduciéndose en un mayor costo tanto de distancias recorridas como económico para la empresa, en términos del bono operativo que se les asigna para los recorridos.

                                ### Objetivo

                                Encontar la ruta de los colaboradores que minimice la distancia recorrida. En las reuniones de seguimiento, el colaborador debe visitar a todos sus clientes y solo los puede visitar una sola vez. Así, el problema es similar al que se tiene con el de **Travel salesman person**.

                                ### Algoritmos

                                Para resolver el problema antes planteado, se revisarán los siguientes algoritmos:

                                - Particle Swarm (PS)
                                - Simulated Annealing (SA)

                                #### Códigos
                                ###### Del proyecto
                                Para aprender más, visita el [repositorio del proyecto](https://github.com/lauragmz/proyecto-final-mno2020/)

                                """
                                )
                            )
                        ],
                    ),
                ],
            )
        ),
    )


######################## START RESULTS ########################

layout = dict(
    autosize=True,
    automargin=True,
    margin=dict(l=30, r=30, b=20, t=40),
    hovermode="closest",
    plot_bgcolor="#F9F9F9",
    paper_bgcolor="#F9F9F9",
    legend=dict(font=dict(size=10), orientation="h"),
    title="Satellite Overview",
)

app.layout = html.Div(
    [
        dcc.Store(id="aggregate_data"),
        # empty Div to trigger javascript file for graph resizing
        html.Div(id="output-clientside"),
        html.Div(
            [
                html.Div(
                    [
                        html.Img(
                            src=app.get_asset_url("logo-ITAM.png"),
                            id="plotly-image",
                            style={
                                "height": "60px",
                                "width": "auto",
                                "margin-bottom": "25px",
                            },
                        )
                    ],
                    className="one-third column",
                ),
                html.Div(
                    [
                        html.Div(
                            [
                                html.H3(
                                    "Problema del vendedor viajero",
                                    style={"margin-bottom": "0px"},
                                ),
                                html.H5(
                                    "Proyecto final Métodos Númericos y Optimización", style={"margin-top": "0px"}
                                ),
                            ]
                        )
                    ],
                    className="one-half column",
                    id="title",
                ),
                html.Div(
                    [
                        html.Button("Más información", id="learn-more-button", n_clicks=0),
                    ],
                    className="one-third column",
                    id="button",
                ),
            ],
            id="header",
            className="row flex-display",
            style={"margin-bottom": "35px"},
        ),

        html.Div(
            [

                html.Div(
                    [html.Iframe(id = "map", srcDoc = open("mapas/mapa_base.html", 'r').read(),
                    width=650, height=350)],
                     className="pretty_container seven columns",
                ),

                html.Div([
                html.P(id="radioitems-checklist-output",
                        className="control_label"),

                html.H6(
                    "Selecciona un algoritmo",
                    className="control_label",
                ),
                dcc.RadioItems(
                    id="tipoAlgoritmo",
                    options=[
                        {"label": "Simulated Annealing", "value": 1},
                        {"label": "Particle Swarm", "value": 2},
                        {"label": "Ambos", "value": 3},
                    ],
                    value=1,
                    #labelStyle={"display": "inline-block"},
                    className="dcc_control",
                ),
                html.H6(
                    "Selecciona un estado:",
                    className="control_label",
                ),
                dcc.Dropdown(
                    id="ciudad_dropdown",
                    #options=edo.to_dict("records"),
                    value="Jalisco",
                    className="dcc_control",
                    multi=False,
                ),

                html.H6(
                    "Selecciona una fuerza de venta",
                    className="control_label",
                ),
                dcc.Dropdown(
                    id="fza_dropdown",
                    #options=edo.to_dict("records"),
                    value=1,
                    className="dcc_control",
                    multi=False,
                ),

                html.P(
                    "",
                    className="control_label",
                ),
                html.Button("Ver ruta", id="button_envia", className="control_center"),


                ],
                className="pretty_container five columns"),

            ],
            className="row flex-display",
            ),


        html.Div(
            [
                html.Div(
                    [html.P(id="ruta-corta"),
                    html.H6("Ruta más corta")],
                    id="wells",
                    className="mini_container",
                ),

                html.Div(
                    [html.H6(id="distancia"), html.P("Distancia (en kilómetros)")],
                    id="esc",
                    className="mini_container",
                ),
            ],
            id="info-container",
            className="row container-display",
        ),
         markdown_popup(),
    ],
    id="mainContainer",
    style={"display": "flex", "flex-direction": "column"},
)








@app.callback(
    Output("fade", "is_in"),
    [Input("fade-button", "n_clicks")],
    [State("fade", "is_in")],
)
def toggle_fade(n, is_in):
    if not n:
        # Button has never been clicked
        return False
    return not is_in


@app.callback(
    [Output("map", "srcDoc"),
    Output('distancia','children'),
    Output('ruta-corta','children')],
    [Input('button_envia', "n_clicks")],
    [State('tipoAlgoritmo', "value"),
     State("fza_dropdown", "value")])

def update_results(n_clicks, value_tipo, value_fza):
    """
    Actualiza la tabla y el mapa
    Obtiene los resultados con las especificaciones del usuario

    """
    if n_clicks is None:
        raise PreventUpdate
    else:
        if value_fza == []:
            raise PreventUpdate

        map_name = "mapas/mapa_"+ str(value_tipo) + \
                    "_" + str(value_fza) + ".html"

        agent = ShortestPath()
        # Si ya existe el mapa, solo saca la distancia
        if os.path.isfile(map_name)==True:
            distancia, ruta = agent.get_distance(value_tipo,value_fza)
        else:
            distancia, ruta = agent.get_path(value_tipo,value_fza,map_name)

        abre_mapa = open(map_name, 'r').read()
        distancia = format(float(round(distancia,2)), ',')

        return abre_mapa, distancia, ruta

@app.callback(
    Output("radioitems-checklist-output", "children"),
    [
        Input("fza_dropdown", "value"),
    ],
)
def on_form_change(n_drop):
    error = "Por favor, selecciona una fuerza de ventas."
    if n_drop == []:
        output_string = error
    else:
        output_string = ""

    return output_string


@app.callback(
    [Output('ciudad_dropdown', 'options')],
    [Input('tipoAlgoritmo', 'value')]
)
def update_dropdown_ciudad(tipo):
    if tipo  ==  int(1):
        algorithm_name = 'SA'
    elif tipo  ==  int(2):
        algorithm_name = 'PS'

    query = "select distinct estado from trabajo.resultados_vw \
    order by estado ; "
    print(query)
    df = get_data(query)
    df.columns = ['label']
    df['value'] = df['label'] #df.index +1
    print(df)
    resp = [df.to_dict("records")]
    return resp



@app.callback(
    [Output('fza_dropdown', 'options')],
    [Input('ciudad_dropdown', 'value')]
)
def update_dropdown_fza(tipo):

    query =  "select distinct id_fza_ventas " + \
    "from trabajo.resultados_vw where estado = '" + \
    str(tipo) +  "' order by id_fza_ventas;"
    print(query)

    df = get_data(query)
    df.columns = ['label']
    df['value'] = df['label']
    resp = [df.to_dict("records")]
    return resp


@app.callback(
    Output("markdown", "style"),
    [Input("learn-more-button", "n_clicks"), Input("markdown_close", "n_clicks")],
)
def update_click_output(button_click, close_click):
    ctx = dash.callback_context
    prop_id = ""
    if ctx.triggered:
        prop_id = ctx.triggered[0]["prop_id"].split(".")[0]

    if prop_id == "learn-more-button":
        return {"display": "block"}
    else:
        return {"display": "none"}


################################# MAIN ################################
if __name__ == '__main__':
    app.run_server(host="0.0.0.0", debug=True)
