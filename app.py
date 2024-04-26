from flask import Flask, render_template, request
import subprocess
import dash
from dash import html, dcc
import pandas

# Existing Flask app
app = Flask(__name__)

# Create Dash app within the existing Flask app
dash_app = dash.Dash(server=app, url_base_pathname='/dashboard/')

# Define Dash layout
dash_app.layout = html.Div(children=[
    html.H1(children='Hello Dash'),
    html.Div(children='Dash: A web application framework for Python.'),
    dcc.Graph(
        id='example-graph',
        figure={
            'data': [
                {'x': [1, 2, 3], 'y': [4, 1, 2], 'type': 'bar', 'name': 'SF'},
                {'x': [1, 2, 3], 'y': [2, 4, 5], 'type': 'bar', 'name': 'Montréal'},
            ],
            'layout': {
                'title': 'Dash Data Visualization'
            }
        }
    )
])

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/train_model', methods=['POST','GET'])
def train_model():
    dataset = request.form['dataset']
    # Replace 'NID.py' with your actual Python script filename
    data = subprocess.run(['python', 'NID.py', dataset], capture_output=True, text=True)
    return render_template('results.html', data=data.stdout)
    # data = [line.split() for line in result.stdout.strip().split('\n') if line.strip()]
    # Ensure each sublist has the same length by padding shorter sublists with empty strings
    # max_len = max(len(sublist) for sublist in data)
    # data = [sublist + ['']*(max_len-len(sublist)) for sublist in data]
    # Render the 'results.html' template with the parsed rows
    

if __name__ == '__main__':
    app.run(debug=True)
