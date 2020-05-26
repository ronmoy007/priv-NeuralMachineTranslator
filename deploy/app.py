import tensorflow as tf

from src.classes import Encoder
from src.classes import Decoder

import unicodedata
import re
import pickle

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
import base64

UPLOAD_DIRECTORY = "files"

if not os.path.exists(UPLOAD_DIRECTORY):
    os.makedirs(UPLOAD_DIRECTORY)

#---------------- INICIALIZE ----------------
print(dcc.__version__) # 0.6.0 or above is required
external_stylesheets = [dbc.themes.BOOTSTRAP]
neuralmachinetranslator = Flask(__name__)
app = dash.Dash(server=neuralmachinetranslator,  meta_tags=[{"name": "viewport", "content": "width=device-width"}])

app.title= "Neural Machine Translation y mecanismos de atención"
app.config.suppress_callback_exceptions = True

image_filename = 'matrizI.png'
encoded_image = base64.b64encode(open(image_filename, 'rb').read())

# open a file, where you stored the pickled data
file = open('./training_checkpoints/nbr_EncDec_Units.pickle', 'rb')
nbr_EncDec_Units = pickle.load(file)
file.close()

file = open('./training_checkpoints/nbr_TamanioMax_FraseInput.pickle', 'rb')
nbr_TamanioMax_FraseInput = pickle.load(file)
file.close()

file = open('./training_checkpoints/nbr_TamanioMax_FraseTarget.pickle', 'rb')
nbr_TamanioMax_FraseTarget = pickle.load(file)
file.close()

file = open('./training_checkpoints/Tokenizador_Frases_Input.pickle', 'rb')
Tokenizador_Frases_Input = pickle.load(file)
file.close()

file = open('./training_checkpoints/Tokenizador_Frases_Target.pickle', 'rb')
Tokenizador_Frases_Target = pickle.load(file)
file.close()

file = open('./training_checkpoints/nbr_TamanioVoc_Input.pickle', 'rb')
nbr_TamanioVoc_Input = pickle.load(file)
file.close()

file = open('./training_checkpoints/nbr_TamanioVoc_Target.pickle', 'rb')
nbr_TamanioVoc_Target = pickle.load(file)
file.close()

file = open('./training_checkpoints/nbr_EmbeddingDim.pickle', 'rb')
nbr_EmbeddingDim = pickle.load(file)
file.close()

file = open('./training_checkpoints/nbr_TamanioBatch.pickle', 'rb')
nbr_TamanioBatch = pickle.load(file)
file.close()

# restoring the latest checkpoint in checkpoint_dir
encoder = Encoder(nbr_TamanioVoc_Input, nbr_EmbeddingDim, nbr_EncDec_Units, nbr_TamanioBatch)
#encoder = Encoder(0, 0, 0, 0)
decoder = Decoder(nbr_TamanioVoc_Target, nbr_EmbeddingDim, nbr_EncDec_Units, nbr_TamanioBatch)
#decoder = Decoder(0, 0, 0, 0)
optimizer = tf.keras.optimizers.Adam()

checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(optimizer=optimizer,
                                 encoder=encoder,
                                 decoder=decoder)
checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))


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
                                    "Métodos analíticos",
                                    style={"margin-bottom": "0px"},
                                ),
                                html.H5(
                                    "Neural Machine Translation y mecanismos de atención", style={"margin-top": "0px"}
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
                    [
                    html.H6("Ingresa el texto a traducir",className="control_label"),
                    html.Div(dcc.Input(id='input-on-submit', type='text', value='Hola a todos')),
                    html.Button('Traducir', id='submit-val', n_clicks=0),
                    html.Div(id='container-button-basic', children='Enter a value and press submit')

                    ],
                    className="pretty_container five columns"
                ),

                html.Div(
                    [
                    html.Img(id="plotly-image2", src='data:image/png;base64,{}'.format(encoded_image.decode()))
                    ],
                    className="pretty_container seven columns",
                ),

            ],
            className="row flex-display",
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

@app.callback(
    [dash.dependencies.Output('container-button-basic', 'children'),
    Output("plotly-image2", "src")],
    [dash.dependencies.Input('submit-val', 'n_clicks')],
    [dash.dependencies.State('input-on-submit', 'value')]
    )
def update_output(n_clicks, value):
    if n_clicks==0:
        #image_filename = 'assets/books_read.png'
        #encoded_image = base64.b64encode(open(image_filename, 'rb').read())
        #str_ValorPlot = 'data:image/png;base64,{}'.format(encoded_image.decode())
        #str_Resultado = ''
        #str_Resultado = func_aux(value)
        str_Resultado = 'Traducción: {}'.format('hi ! <end>')
        encoded_image = base64.b64encode(open('matrizI.png', 'rb').read())
        str_ValorPlot = 'data:image/png;base64,{}'.format(encoded_image.decode())
    else:
        if str(value) == 'None' or str(value) == '':
            str_Resultado = 'Se debe capturar algún texto'
        else:
            try:
                str_Resultado = func_aux(str(value))
                encoded_image = base64.b64encode(open('matrizR.png', 'rb').read())
                str_ValorPlot = 'data:image/png;base64,{}'.format(encoded_image.decode())
            except BaseException as errorPrueba:
                str_Resultado = 'La siguiente palabra no se encuentra en el diccionario: ' + str(errorPrueba)
                encoded_image = base64.b64encode(open('matrizI.png', 'rb').read())
                str_ValorPlot = 'data:image/png;base64,{}'.format(encoded_image.decode())

        #import matplotlib.pyplot as plt
        #plt.plot([0, 1, 2, 3, 4], [0, 3, 5, 9, 11])
        #plt.xlabel('Months')
        #plt.ylabel('Books Read'+ str_Resultado)
        #plt.savefig('books_read.png')

        #encoded_image = base64.b64encode(open('books_read.png', 'rb').read())
        #str_ValorPlot = 'data:image/png;base64,{}'.format(encoded_image.decode())

    return str_Resultado, str_ValorPlot


def func_aux(str_Input):
    # str_Resultado = 'Este es el resultado de la función' + str_Input
    print(nbr_EncDec_Units)
    result, sentence, np_attention_plot = evaluate(str_Input)

    np_attention_plot = np_attention_plot[:len(result.split(' ')), :len(sentence.split(' '))]
    plot_attention(np_attention_plot, sentence.split(' '), result.split(' '))

    return 'Traducción: {}'.format(result)

def ConvertirUnicodeToAscii(str_Texto):
    return ''.join(c for c in unicodedata.normalize('NFD', str_Texto) if unicodedata.category(c) != 'Mn')


def PrepararOraciones(str_Oracion):
    str_Oracion = ConvertirUnicodeToAscii(str_Oracion.lower().strip())

    # Se crea un espacio entre las palabras y signos de puntuación
    str_Oracion = re.sub(r"([?.!,¿])", r" \1 ", str_Oracion)
    str_Oracion = re.sub(r'[" "]+', " ", str_Oracion)

    # Se reemplaza todo por un espacio excepto letras y signos especificados
    str_Oracion = re.sub(r"[^a-zA-Z?.!,¿]+", " ", str_Oracion)

    str_Oracion = str_Oracion.strip()

    # Se agrega el token de inicio y fin
    str_Oracion = '<start> ' + str_Oracion + ' <end>'
    return str_Oracion

def evaluate(str_Oracion):
    np_attention_plot = np.zeros((nbr_TamanioMax_FraseTarget, nbr_TamanioMax_FraseInput))

    str_OracionPreparada = PrepararOraciones(str_Oracion)

    inputs = [Tokenizador_Frases_Input.word_index[i] for i in str_OracionPreparada.split(' ')]
    inputs = tf.keras.preprocessing.sequence.pad_sequences([inputs],
                                                         maxlen=nbr_TamanioMax_FraseInput,
                                                         padding='post')
    tensor_Input = tf.convert_to_tensor(inputs)

    result = ''

    tensor_Encoder_Oculto_Init = [tf.zeros((1, nbr_EncDec_Units))]

    tensor_Encoder_Output, tensor_Encoder_Estado_Oculto = encoder(tensor_Input, tensor_Encoder_Oculto_Init)

    tensor_Decoder_Estado_Oculto = tensor_Encoder_Estado_Oculto
    tensor_Decoder_Input = tf.expand_dims([Tokenizador_Frases_Target.word_index['<start>']], 0)

    # nbr_TamanioMax_FraseTarget, nbr_TamanioMax_FraseInput
    for token in range(nbr_TamanioMax_FraseTarget):
        # tensor_decoder_output, tensor_state, tensor_attention_weights
        tensor_Predicciones, tensor_Decoder_Estado_Oculto, tensor_attention_weights = decoder.call(tensor_Decoder_Input,
                                                             tensor_Decoder_Estado_Oculto,
                                                             tensor_Encoder_Output)

        # storing the attention weights to plot later on
        tensor_attention_weights = tf.reshape(tensor_attention_weights, (-1, ))
        np_attention_plot[token] = tensor_attention_weights.numpy()

        predicted_id = tf.argmax(tensor_Predicciones[0]).numpy()

        result += Tokenizador_Frases_Target.index_word[predicted_id] + ' '

        if Tokenizador_Frases_Target.index_word[predicted_id] == '<end>':
            return result, str_OracionPreparada, np_attention_plot

        # the predicted ID is fed back into the model
        tensor_Decoder_Input = tf.expand_dims([predicted_id], 0)

    return result, str_OracionPreparada, np_attention_plot


def plot_attention(attention, sentence, predicted_sentence):
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(1, 1, 1)
    ax.matshow(attention, cmap='viridis')

    fontdict = {'fontsize': 14}

    ax.set_xticklabels([''] + sentence, fontdict=fontdict, rotation=90)
    ax.set_yticklabels([''] + predicted_sentence, fontdict=fontdict)

    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    plt.savefig('matrizR.png', dpi=65)

################################# MAIN ################################
if __name__ == '__main__':
    app.run_server(host="0.0.0.0", debug=True)
