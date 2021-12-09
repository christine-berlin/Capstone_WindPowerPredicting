from io import SEEK_CUR
import pandas as pd
import numpy as np

import dash
#import dash_core_components as dcc
from dash import dcc
#import dash_html_components as html
from dash import html
from dash import dash_table
import plotly.graph_objects as go
import plotly.express as px

import dash_bootstrap_components as dbc
import dash_html_components as html


from dash.dependencies import Input, Output, State

from predict_api import make_prediction



external_stylesheets = ["https://codepen.io/chriddyp/pen/bWLwgP.css"]

################################################################################
# HELPER FUNCTIONS
################################################################################
def degrees_to_cardinal(d):
    dirs = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE",
            "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"]
    ix = int((d + 11.25)/22.5)
    return dirs[ix % 16]

# df['WD100CARD'] = df.WD100.apply(lambda x: degrees_to_cardinal(x))
# df['WD10CARD'] = df.WD10.apply(lambda x: degrees_to_cardinal(x))

def card_sorter(column):
    """Sort function"""
    wd_card=["N","NNE","NE","ENE","E","ESE", "SE", "SSE","S","SSW","SW","WSW","W","WNW","NW","NNW"]
    correspondence = {card: order for order, card in enumerate(wd_card)}
    return column.map(correspondence)




################################################################################
# APP INITIALIZATION
################################################################################
#app = dash.Dash('Wind energy forecast', external_stylesheets=external_stylesheets)
app = dash.Dash(external_stylesheets=[dbc.themes.BOOTSTRAP])


# this is needed by gunicorn command in procfile
server = app.server


################################################################################
# PLOTS
################################################################################

## get color scheme right
colors = px.colors.qualitative.Plotly
color_dict = {'Zone '+str(z): c for z,c in zip(range(1,11), colors)}

## figure energy output over 24 h
def get_figure_24h(dff, selected_zone, selected_hour=0):
    tmin, tmax = 0,24
    if 'HOUR' in dff.columns:
        tmin = dff['HOUR'].min()
        tmax = dff['HOUR'].max()

    fig = go.Figure()
    selected_zone = sorted(selected_zone, key=lambda x : x[-2:])
    # plot the zones
    for column in selected_zone:
        color = color_dict[column]
        fig.add_traces(go.Scatter(x=dff['HOUR'], y = df[column], 
                    mode = 'lines', line=dict(color=color), name=column))
    # plot a line to 
    fig.add_shape(type="line",
        x0=selected_hour, y0=0, x1=selected_hour, y1=1,
        line=dict(
            color="red",
            width=1,
            dash="dash",
        )
    )
    fig.update_xaxes(range = [tmin, tmax])
    fig.update_yaxes(range = [0,1])
    fig.layout.template = 'plotly_white'
    fig.layout.showlegend = True
    return fig

## energy output at certain hour
def get_figure_energy_per_hour(df, selected_zone, selected_hour):
    selected_zone = sorted(selected_zone, key=lambda x : x[-2:])
    df_hour = df[selected_zone]
    df_hour = pd.DataFrame(df_hour.loc[selected_hour:selected_hour])
    cols = df_hour.columns
    cols = [cc for cc in cols if cc.startswith('Zone')]
    dff = df_hour[cols]
    dff = dff.T
    # dff.reset_index(inplace=True)
    bars = []
    fig = go.Figure()
    for zone in selected_zone:
        color = color_dict[zone]
        fig.add_traces(
            go.Bar(x=[zone], y=[dff.loc[zone][dff.columns[-1]]], 
                marker={'color': color}, showlegend=False)
        )
    fig.update_yaxes(range = [0,1])
    fig.layout.template = 'plotly_white'
    return fig

## wind rose
def plot_windrose(df, selected_zone=1):
    df = df[df['ZONEID']==selected_zone]
    bins = np.linspace(0,24,13)
    labels = [0,2,4,6,8,10,12,14,16,18,20,22]

    df['WS100BIN'] = pd.cut(df['WS100'], bins=bins, labels = labels)

    df_grouped = df.groupby(['WD100CARD','WS100BIN']).count().reset_index()
    
    wd_card=["N","NNE","NE","ENE","E","ESE", "SE", "SSE","S","SSW","SW","WSW","W","WNW","NW","NNW"]

    wd_zeros = np.zeros(len(wd_card))
    df_all_wd_card = pd.DataFrame([wd_card, wd_zeros, wd_zeros])
    df_all_wd_card = df_all_wd_card.T
    df_all_wd_card.columns = ['WD100CARD', 'WS100BIN', 'FREQUENCY']
    
    data_wind = df_grouped[['WD100CARD', 'WS100BIN', 'TIMESTAMP']]
    data_wind.columns = df_all_wd_card.columns
    #print(data_wind.head(50))

    datax = pd.concat([data_wind, df_all_wd_card], axis = 0)
    datax = datax.sort_values(by='WD100CARD', key=card_sorter)
    ws_ls = np.linspace(0,25,26)
    wind_all_speeds = pd.DataFrame([['N']*len(ws_ls), ws_ls, np.zeros(len(ws_ls))]).T
    wind_all_speeds.columns = ['WD100CARD', 'WS100BIN', 'FREQUENCY']
    wind_all_speeds
    datax = pd.concat([wind_all_speeds, df_all_wd_card, data_wind], axis = 0)

    print('Wind data\n',datax.head())

    fig = px.bar_polar(datax, 
                   r="FREQUENCY", 
                   theta="WD100CARD",
                   color="WS100BIN", 
                   #template="plotly_black",
                   color_discrete_sequence= px.colors.sequential.Plasma_r,
                   )                     
    fig.layout.showlegend = True
    fig.update_layout(
            autosize=True,
            # width=250,
            # height=250,
            )

    #fig.show()
    return fig


################################################################################
# LAYOUT
################################################################################
day='2013-01-01'
filename = 'api/prediction_for_dashboard.csv'
filename_wind = 'api/prediction_wind_for_dashboard.csv'
# df, df_wind = make_prediction(day)
# df['HOUR'] = df['TIMESTAMP'].dt.hour
# df_wind['WD100CARD'] = df_wind.WD100.apply(lambda x: degrees_to_cardinal(x))
# print(df_wind.head())
# df.to_csv(filename,index=False)
# df_wind.to_csv(filename_wind,index=False)
df = pd.read_csv(filename)
df_wind = pd.read_csv(filename_wind)

title_subtitle = html.Div( 
            [
                html.H2(
                    id="title",
                    children="Wind energy forecast",
                ),
                html.H3(
                    id="subtitle",
                    children="Forecast of the power output over the next 24 hours.",
                )
            ]
        )
zone_selector = html.Div( 
            dbc.Card(
                [
                    dbc.CardBody(
                        [ dcc.Checklist( 
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
                        ]
                    )
                ]
            )
        )
                            
graph_24h = html.Div( 
                        [html.H3('Energy output over 24 h'),
                        dcc.Graph(id="plot-24h")],
                        className="six columns"
                    )
graph_energy_per_hour = html.Div( 
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
                        ],className="six columns"
                    )
dropdown_wind_rose = dcc.Dropdown(
        id='dropdown_windrose',
        options=[
            {'label': 'Zone 1', 'value': 1},
            {'label': 'Zone 2', 'value': 2},
            {'label': 'Zone 3', 'value': 3},
            {'label': 'Zone 4', 'value': 4},
            {'label': 'Zone 5', 'value': 5},
            {'label': 'Zone 6', 'value': 6},
            {'label': 'Zone 7', 'value': 7},
            {'label': 'Zone 8', 'value': 8},
            {'label': 'Zone 9', 'value': 9},
            {'label': 'Zone 10', 'value': 10},
        ],
        value='Zone 1',
        placeholder="Select zone for wind rose.",

    ),
graph_wind_rose = html.Div( 
                        [
                            dbc.Row(
                                [
                                    dbc.Col(dropdown_wind_rose, width="4"),
                                ]
                            ),
                            dbc.Row(
                                [
                                    dbc.Col([dcc.Graph(id='windrose')], width="8"), 
                                    
                                ]
                            ),
                        ]
                    )



app.layout = html.Div( 
    [ 
        dbc.Row(dbc.Col([title_subtitle], width=12)),
        dbc.Row(
            [
                dbc.Col([zone_selector], width=2),
                dbc.Col([graph_wind_rose], width=10),
            ]
        ),
        dbc.Row(dbc.Col([graph_24h], width="10")),
        dbc.Row(dbc.Col([graph_energy_per_hour], width="10")),
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
    Input('zone-selector', 'value'),
    Input('energy-per-hour-slider', 'value'))
def update_graphs(selected_zone, selected_hour):
    cols = selected_zone.copy()
    cols.append('TIMESTAMP')
    

    return get_figure_24h(df, selected_zone, selected_hour)


@app.callback(
    Output('energy-per-hour-figure', 'figure'),
    Input('zone-selector', 'value'),
    Input('energy-per-hour-slider', 'value'))
def update_figure(selected_zone, selected_hour):
    print('in update_figure')
    print('selected_zone', selected_zone)
    print('selected_hour', selected_hour)
    
    return get_figure_energy_per_hour(df, selected_zone, selected_hour)


@app.callback( 
    Output('windrose', 'figure'),
    Input('dropdown_windrose', 'value')
    )
def update_windrose(selected_zone):
    print('selected_zon ein update_windrose',selected_zone)
    return plot_windrose(df_wind, selected_zone)





# Add the server clause:
if __name__ == "__main__":
    app.run_server()



# @app.callback( 
#     Output('windrose', 'x'),
#     Input('zone-selector', 'value')
#     )
# def update_windrose(selected_zone):
#     return plot_windrose(df_wind, selected_zone)



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