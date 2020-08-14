import pickle

import numpy as np

import dash
import dash_html_components as html
import dash_core_components as dcc
from dash.dependencies import Input, Output
import shap
import xgboost

from graphlib import donut_chart, waterfall_chart

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

filename_path = '../pottery/training/output/beastML.pkl'

# print(type(model))
genders = {"Female": 0, "Male": 1, }
educations = {"None": 0, "Lower than High School": 1, "High-School/GED": 2, "College/eqv": 3, "Graduate": 4}
ages = {"Under 35": 0, "35-55": 1, "Over 55": 2}
socio = {"Bottom 0-10%": 5, "Bottom 10-20%": 15, "Bottom 20-30%": 25, "Bottom 30-40%": 35, "Top 40-50%": 45,
           "Top 50-60%": 55, "Top 60-70%": 65, "Top 70-80%": 75,  "Top 80-90%": 85, "Top 90-100%": 95}
disability = {"No": 0, "Yes": 1}
productivity = {"I am Yoda": 250, "Work hard, play-hard": 170, "I take things easy": 155,
                "Cs get degrees, right?": 140, "I like to sleep": 100}


app.layout = html.Div([

    html.Div([

        html.Div([
            html.H6("Visualizing BlackBox ML Models"),
            html.H6("We predict student's will succeed based on these attributes"),
            html.Div(["Gender: ", dcc.Dropdown(id='gender',
                                               options=[{'label': k, 'value': v} for k, v in genders.items()],
                                               value=1)]),
            html.Br(),
            html.Div(["Highest Education: ", dcc.Dropdown(id='education',
                                                          options=[{'label': k, 'value': v} for k, v in educations.items()],
                                                          value=2,)]),
            html.Br(),
            html.Div(["Age: ", dcc.Dropdown(id='age',
                                            options=[{'label': k, 'value': v} for k, v in ages.items()],
                                            value=0, )]),
            html.Br(),
            html.Div(["Socio-Economic-Range: ", dcc.Dropdown(id='soecon',
                                            options=[{'label': k, 'value': v} for k, v in socio.items()],
                                            value=55.0, )]),
            html.Br(),
            html.Div(["Num of Prev Attempts (for course you'll be taking): ", dcc.Input(id='attempt', value=0, type='number')]),
            html.Br(),
            html.Div(["Credits (30-300): ", dcc.Input(id='credit', value='30', type='number')]),
            html.Br(),
            html.Div(["Disability: ", dcc.Dropdown(id='disability',
                                               options=[{'label': k, 'value': v} for k, v in disability.items()],
                                               value=0)]),
            html.Br(),
            html.Div(["How Productive are you?: ", dcc.Dropdown(id='productivity',
                                               options=[{'label': k, 'value': v} for k, v in productivity.items()],
                                               value=170)]),
            html.Br(),
            html.H6(children=[html.Div(id='my-output',)]),

        ], id='control_panel'),

        html.Div([

            dcc.Graph(id='id_donut'),
            dcc.Graph(id='id_waterfall')

        ], id='graph_output_panel')

    ], className="dash-bg-body"),

])


@app.callback(
    [Output(component_id='my-output', component_property='children'),
     Output(component_id='id_donut', component_property='figure'),
     Output(component_id='id_waterfall', component_property='figure')],

    [Input(component_id='gender', component_property='value'),
     Input(component_id='education', component_property='value'),
     Input(component_id='soecon', component_property='value'),
     Input(component_id='age', component_property='value'),
     Input(component_id='attempt', component_property='value'),
     Input(component_id='credit', component_property='value'),
     Input(component_id='disability', component_property='value'),
     Input(component_id='productivity', component_property='value')]
)
def update_output_div(gender, education, soecon, age, attempt, credit, disability, productivity):
    # input = np.array([1, 3, 95., 2, 0, 240, 0, 120.]).reshape(1, -1)
    input = np.array([gender, education, soecon, age, attempt, credit, disability, productivity]).reshape(1, -1)
    model = pickle.load(open(filename_path, 'rb'))
    mybooster = model.get_booster()

    model_bytearray = mybooster.save_raw()[4:]

    def myfun(self=None):
        return model_bytearray

    mybooster.save_raw = myfun
    explainer = shap.TreeExplainer(mybooster, feature_perturbation="interventional")
    shap_values = explainer.shap_values(input)
    base_value = explainer.expected_value / len(input[0])
    print(shap_values)
    total = np.sum(np.abs(shap_values))
    percents = []
    # sum over each feature
    pred = model.predict(input)[0]
    if pred == 1:
        pred = "passed"
    else:
        pred = "failed"
    fi_shap = abs(shap_values).sum(0)

    # Normalize
    fi_shap = fi_shap / fi_shap.sum();
    fi_shap.sum(), fi_shap.shape

    # odds = np.exp(shap_values)
    # prob = odds / (1 - odds)
    return f"Predicting Student Outcome As: '{pred}'", donut_chart(shap_values, pred), \
           waterfall_chart(fi_shap)


if __name__ == '__main__':
    app.run_server(debug=True)