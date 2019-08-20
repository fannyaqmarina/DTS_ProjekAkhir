
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


df1 = pd.read_csv(r"C:\Users\ASPIRE E 14\Music\DTS FGA 2019 - Unmul\projek akhir\ProjekAkhir\diabetes.csv")
data = df1['Diabetic'].value_counts()

logistik = 'Logistik.png'
KNN = 'KNN.png'
knnNotNorm = 'KNNnotNorm.png'
KNNNorm='KNNNorm.png'
Histogram = 'Histogram.png'
test_logistik = base64.b64encode(open(logistik, 'rb').read()).decode('ascii')
test_histogram = base64.b64encode(open(Histogram, 'rb').read()).decode('ascii')
test_Knn = base64.b64encode(open(KNN, 'rb').read()).decode('ascii')
test_Knnnotnorm = base64.b64encode(open(knnNotNorm, 'rb').read()).decode('ascii')
test_Knnnorm = base64.b64encode(open(KNNNorm, 'rb').read()).decode('ascii')

#DASH APP
app = dash.Dash()
colors = {
    'background': '#ffffff',
    'text': '#ff6977'
}
app.layout = html.Div(style={'backgroundColor': colors['background']}, children=[
    html.H1(
        children='Dash Kelompok 5',
        style={
            'textAlign': 'center',
            'color': colors['text']
        }
    ),
    html.Div(children='Klasifikasi Penyakit Diabetes Berdasarkan Tindakan Diagnostik', style={
        'textAlign': 'center',
        'color': colors['text']
    }),
     dcc.Graph(
        id='Graph1',
        figure={
            'data': [
                {'x': [0,1], 'y': [data[0],data[1]], 'type': 'bar'},
            ],
            'layout': {
                'plot_bgcolor': colors['background'],
                'paper_bgcolor': colors['background'],
                'font': {
                    'color': colors['text']
                }
            }
        }
    ),
      html.H2('Histogram',style={
            'textAlign': 'center',
            'color': colors['text']}),
    html.Img(src='data:image/png;base64,{}'.format(test_histogram),style={
            'textAlign': 'center',
            'color': colors['text']}),
     html.H2('K-Nearest Neighbor Sebelum Normalisasi',style={
            'textAlign': 'center',
            'color': colors['text']}),
    html.Img(src='data:image/png;base64,{}'.format(test_Knnnotnorm),style={
            'textAlign': 'center',
            'color': colors['text']}),
     html.H2('K-Nearest Neighbor Setelah Normalisasi',style={
            'textAlign': 'center',
            'color': colors['text']}),
    html.Img(src='data:image/png;base64,{}'.format(test_Knnnorm),style={
            'textAlign': 'center',
            'color': colors['text']}),
     html.H2('K-Nearest Neighbor Akurasi',style={
            'textAlign': 'center',
            'color': colors['text']}),
    html.Img(src='data:image/png;base64,{}'.format(test_Knn),style={
            'textAlign': 'center',
            'color': colors['text']}),
    html.H2('Regresi Logistik',style={
            'textAlign': 'center',
            'color': colors['text']}),
    html.Img(src='data:image/png;base64,{}'.format(test_logistik),style={
            'Align': 'center',
            'color': colors['text']}),
   
])
            
if __name__ == '__main__':
    app.run_server(debug=False)