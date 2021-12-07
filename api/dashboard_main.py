import pandas as pd

import dash
#import dash_core_components as dcc
from dash import dcc
#import dash_html_components as html
from dash import html
from dash import dash_table
import plotly.graph_objects as go
from dash.dependencies import Input, Output, State

from predict_api import make_prediction



external_stylesheets = ["https://codepen.io/chriddyp/pen/bWLwgP.css"]


################################################################################
# APP INITIALIZATION
################################################################################
app = dash.Dash('Wind energy forecast', external_stylesheets=external_stylesheets)

# this is needed by gunicorn command in procfile
server = app.server


################################################################################
# PLOTS
################################################################################
LEGEND = ["clicks", "go fish!"]
SCORES = [0.1, 0.1]


def get_figure(legend, scores):
    return go.Figure(
        [go.Bar(x=legend, y=scores)],
        layout=go.Layout(template="simple_white"),
    )
fig = get_figure(LEGEND, SCORES)

def get_figure_24h(dff, selected_columns):
    if len(selected_columns) == 0:
        return None
    else:
        if len(selected_columns) == 0:
            data = []
        else:
            data = [{
                        "x": dff["TIMESTAMP"],
                        "y": dff[zz],
                        "type": "line",
                        "marker": {"color": 'green'},} for zz in selected_columns
                        ]
        return [
        dcc.Graph(
            #id=column,
            figure={
                "data": data,
                "layout": {
                    "xaxis": {"automargin": True},
                    "yaxis": {
                        "automargin": True,
                        "title": {"text": 'Zone 1'}
                    },
                    "height": 250,
                    "margin": {"t": 10, "l": 10, "r": 10},
            },  },
        )
    ]

dff = pd.DataFrame()
selected_columns = []
figure_24h = get_figure_24h(dff, selected_columns)


################################################################################
# LAYOUT
################################################################################
day='2013-01-01'
df = make_prediction(day)

app.layout = html.Div(
    [
        html.H2(
            id="title",
            children="Wind energy forecast",
        ),
        html.H3(
            id="subtitle",
            children="Forecast of the power output over the next 24 hours.",
        ),
        html.Div(  # Define the row element
                [
                    dcc.Graph(id="plot_lines"),  # Define the left element
                    dcc.Graph(id="bar-chart"),
                    ],
                    style={'width': '48%', 'display': 'inline-block'}),
        html.Div([
    dash_table.DataTable(
        id='datatable-interactivity',
        columns=[
            {"name": i, "id": i, "deletable": False, "selectable": True} for i in df.columns
        ],
        data=df.to_dict('records'),
        editable=True,
        filter_action="native",
        sort_action="native",
        sort_mode="multi",
        column_selectable="multi",
        row_selectable="single",
        row_deletable=False,
        selected_columns=[],
        selected_rows=[],
        page_action="native",
        page_current= 0,
        page_size= 24,
    )#,
    #html.Div(id='datatable-interactivity-container')
])
        ])


################################################################################
# INTERACTION CALLBACKS
################################################################################
# https://dash.plotly.com/basic-callbacks
@app.callback(
    Output('datatable-interactivity', 'style_data_conditional'),
    Input('datatable-interactivity', 'selected_columns')
)
def update_output(selected_columns):
    return [{
        'if': { 'column_id': i },
        'background_color': '#D2F3FF'
    } for i in selected_columns]


@app.callback(
    Output('plot_lines', "children"),
    Input('datatable-interactivity', 'selected_columns')
)
def update_graphs(selected_columns):
    dff = pd.DataFrame()
    if selected_columns is None:
        dff = df
    else:
        cols = selected_columns
        cols.append('TIMESTAMP')
        dff = df[cols]

    return get_figure_24h(dff, selected_columns)




    # fig = get_figure(LEGEND, SCORES)
    # if n_clicks > 0:
    #     if 0 < len(value) < 10:
    #         text = "you said: " + value
    #         scores = [0.1 * n_clicks, 0.1]
    #         fig = get_figure(LEGEND, scores)
    #         return text, fig
    #     else:
    #         return "Please add a text between 0 and 10 characters!", fig
    # else:
    #     return "", fig
# @app.callback(
#     Output('datatable-interactivity-container', "children"),
#     Input('datatable-interactivity', "derived_virtual_data"),
#     Input('datatable-interactivity', "derived_virtual_selected_rows"))
# def update_graphs(rows, derived_virtual_selected_rows):
    # When the table is first rendered, `derived_virtual_data` and
    # `derived_virtual_selected_rows` will be `None`. This is due to an
    # idiosyncrasy in Dash (unsupplied properties are always None and Dash
    # calls the dependent callbacks when the component is first rendered).
    # So, if `rows` is `None`, then the component was just rendered
    # and its value will be the same as the component's dataframe.
    # Instead of setting `None` in here, you could also set
    # `derived_virtual_data=df.to_rows('dict')` when you initialize
    # the component.
    # print('columns',derived_virtual_selected_rows)
    # print('columns1',rows)
    # if derived_virtual_selected_columns is None:
    #     derived_virtual_selected_columns = []

    # dff = df if rows is None else pd.DataFrame(columns)
    # dff = df if columns is None else pd.DataFrame(co)
    # print(dff.head())

    # colors = ['#7FDBFF' if i in derived_virtual_selected_rows else '#0074D9'
    #           for i in range(len(dff))]

    # graph = [ 
    #     dcc.Graph( 
    #         id=row
    #     )


    # ]
    # return None
    # return [
    #     dcc.Graph(
    #         id=column,
    #         figure={
    #             "data": [
    #                 {
    #                     "x": dff["country"],
    #                     "y": dff[column],
    #                     "type": "bar",
    #                     "marker": {"color": colors},
    #                 }
    #             ],
    #             "layout": {
    #                 "xaxis": {"automargin": True},
    #                 "yaxis": {
    #                     "automargin": True,
    #                     "title": {"text": column}
    #                 },
    #                 "height": 250,
    #                 "margin": {"t": 10, "l": 10, "r": 10},
    #             },
    #         },
    #     )
    #     # check if column exists - user may have deleted it
    #     # If `column.deletable=False`, then you don't
    #     # need to do this check.
    #     for column in ["pop", "lifeExp", "gdpPercap"] if column in dff
    # ]


# Add the server clause:
if __name__ == "__main__":
    app.run_server()
