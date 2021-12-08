from io import SEEK_CUR
import pandas as pd

import dash
#import dash_core_components as dcc
from dash import dcc
#import dash_html_components as html
from dash import html
from dash import dash_table
import plotly.graph_objects as go
import plotly.express as px

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

## get color scheme right
colors = px.colors.qualitative.Plotly
color_dict = {'Zone '+str(z): c for z,c in zip(range(1,11), colors)}

def get_figure_24h(dff, selected_columns):
    tmin, tmax = 0,24
    if 'TIMESTAMP' in dff.columns:
        tmin = dff['TIMESTAMP'].min()
        tmax = dff['TIMESTAMP'].max()
    print('tmin, tmax',tmin,tmax)

    colors = px.colors.qualitative.Plotly
    fig = go.Figure()
    selected_columns = sorted(selected_columns, key=lambda x : x[-2:])
    for column in selected_columns:
        color = color_dict[column]
        fig.add_traces(go.Scatter(x=dff['TIMESTAMP'], y = df[column], 
                    mode = 'lines', line=dict(color=color), name=column))
    # fig.update_xaxes(range = [tmin, tmax])
    fig.update_yaxes(range = [0,1])
    fig.layout.template = 'plotly_white'
    fig.layout.showlegend = True
    return fig

dff = pd.DataFrame()
selected_columns = []
figure_24h = get_figure_24h(dff, selected_columns)

def get_figure_energy_per_hour(df, selected_zone, selected_hour):
    print('In get_figure_energy_per_hour')
    print('selected_zone', selected_zone)
    print('selected_hour', selected_hour)
    print('df.columns', df.columns)
    if selected_zone is None or len(selected_zone)==0:
        return
    selected_zone = sorted(selected_zone, key=lambda x : x[-2:])
    df_hour = df[selected_zone]
    df_hour = pd.DataFrame(df_hour.loc[selected_hour:selected_hour])
    cols = df_hour.columns
    print('cols',cols)
    cols = [cc for cc in cols if cc.startswith('Zone')]
    dff = df_hour[cols]
    dff = dff.T
    # dff.reset_index(inplace=True)
    print('dff.head()\n',dff.head())
    bars = []
    fig = go.Figure()
    for zone in selected_zone:
        color = color_dict[zone]
        print('color', color)
        print('zone', zone)
        print('dff.loc[zone][dff.columns[-1]', dff.loc[zone][dff.columns[-1]])
        fig.add_traces(
            go.Bar(x=[zone], y=[dff.loc[zone][dff.columns[-1]]], 
                marker={'color': color})
        )
    
    fig.update_yaxes(range = [0,1])
    fig.layout.template = 'plotly_white'
    return fig


################################################################################
# LAYOUT
################################################################################
day='2013-01-01'
filename = 'prediction_for_dasboard.csv'
# df = make_prediction(day)
# df['HOUR'] = df['TIMESTAMP'].dt.hour
# df.to_csv(filename,index=False)
df = pd.read_csv(filename)

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
            html.Div( 
                [ 
                    dcc.Checklist( 
                        id="zone-selector",
                        options=[ 
                            {'label': 'Zone 1', 'value': 'Zone 1'},
                            {'label': 'Zone 2', 'value': 'Zone 2'},
                            {'label': 'Zone 3', 'value': 'Zone 3'},
                            {'label': 'Zone 4', 'value': 'Zone 4'},
                            {'label': 'Zone 5', 'value': 'Zone 5'},
                            {'label': 'Zone 6', 'value': 'Zone 6'},
                            {'label': 'Zone 7', 'value': 'Zone 7'},
                            {'label': 'Zone 8', 'value': 'Zone 8'},
                            {'label': 'Zone 9', 'value': 'Zone 9'},
                            {'label': 'Zone 10', 'value': 'Zone 10'}
                        ],
                        value=[],
                        style={'display':'inline'}
                    )
                ], style={"width": '10%', 'display':'block'}
            ),
            html.Div(  # Define the row element
                [
                    html.Div( 
                        [html.H3('Energy output over 24 h'),
                        dcc.Graph(id="plot-24h")],
                        className="six columns"
                    ), 
                    html.Div( 
                        [html.H3('Energy output per hour'),
                        dcc.Graph(id="energy-per-hour-figure"),
                        dcc.Slider( 
                            id='energy-per-hour-slider',
                            min=df['HOUR'].min(),
                            max=df['HOUR'].max(),
                            value=df['HOUR'].min(),
                            marks={str(hh): str(hh) for hh in df['HOUR'].unique()},
                            step=None
                            )
                        ],
                        className="six columns"
                    )
                ],
                className="row"
            ),
            html.Div(
                [
                    dash_table.DataTable(
                        id='datatable-interactivity',
                        columns=[
                            {"name": i, 
                            "id": i, 
                            "deletable": False, 
                            "selectable": True} for i in df.columns
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
                ] 
            )
        ]
    )


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
    Output('plot-24h', "figure"),
    Input('zone-selector', 'value')
)
def update_graphs(selected_columns):
    cols = selected_columns.copy()
    cols.append('TIMESTAMP')
    dff = df[cols]

    return get_figure_24h(dff, selected_columns)


@app.callback(
    Output('energy-per-hour-figure', 'figure'),
    Input('zone-selector', 'value'),
    Input('energy-per-hour-slider', 'value'))
def update_figure(selected_zone, selected_hour):
    print('in update_figure')
    print('selected_zone', selected_zone)
    print('selected_hour', selected_hour)
    
    return get_figure_energy_per_hour(df, selected_zone, selected_hour)






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
