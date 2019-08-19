
import dash
from dash.dependencies import Input, Output
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
import base64
import plotly.graph_objs as go
from scipy.stats import pearsonr as p
from scipy.stats import spearmanr as s
import scipy.stats as stats
import numpy as np


logistik = 'Logistik.png'
KNN = 'KNN.png'
test_logistik = base64.b64encode(open(logistik, 'rb').read()).decode('ascii')
test_Knn = base64.b64encode(open(KNN, 'rb').read()).decode('ascii')


#DASH APP
app = dash.Dash()

app.layout = html.Div(children=[
    html.H2('Regresi Logistik',style={
            'textAlgin':'center'}),
    html.Br(),
    html.Img(src='data:image/png;base64,{}'.format(test_logistik),style={
            'textAlgin':'center'}),
    html.Br(),
     html.Br(),
    html.H2('K-Nearest Neighbor',style={
            'textAlgin':'center'}),
    html.Img(src='data:image/png;base64,{}'.format(test_Knn),style={
            'textAlgin':'center'}),
])
            
if __name__ == '__main__':
    app.run_server(debug=False)