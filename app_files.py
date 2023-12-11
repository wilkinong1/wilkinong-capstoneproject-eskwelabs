import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import ast
from datetime import datetime, timedelta

sample_data = pd.read_csv('./app_sample_data.csv')
results = pd.read_csv('./results_for_season.csv')
upcoming = pd.read_csv('./app_upcoming.csv')
team_logos = {}


underdog_cover = pd.DataFrame({
    'Underdog Covers': [len(sample_data[(sample_data['hcp_tipoff'] > 0) & (sample_data['home_cover_spread'] == 1)]) + len(sample_data[(sample_data['hcp_tipoff'] < 0) & (sample_data['home_cover_spread'] == 0)])],
    'Favorite Covers': [len(sample_data[(sample_data['hcp_tipoff'] < 0) & (sample_data['home_cover_spread'] == 1)]) + len(sample_data[(sample_data['hcp_tipoff'] > 0) & (sample_data['home_cover_spread'] == 0)])]
    })
underdog_cover = underdog_cover.T.reset_index(names='Description')

sample_data['home_cover_spread'] = sample_data['home_cover_spread'].apply(lambda x: "Spread Covered" if x==1 else "Spread Not Covered")
results_full = results.copy(deep=True)

team_list = ['Los Angeles Lakers',
 'Milwaukee Bucks',
 'Indiana Pacers',
 'New Orleans Pelicans',
 'Minnesota Timberwolves',
 'Miami Heat',
 'Orlando Magic',
 'Toronto Raptors',
 'Denver Nuggets',
 'Washington Wizards',
 'Cleveland Cavaliers',
 'Philadelphia 76ers',
 'Golden State Warriors',
 'Chicago Bulls',
 'San Antonio Spurs',
 'Portland Trail Blazers',
 'Oklahoma City Thunder',
 'Brooklyn Nets',
 'LA Clippers',
 'Atlanta Hawks',
 'Charlotte Hornets',
 'Memphis Grizzlies',
 'Utah Jazz',
 'Detroit Pistons',
 'Dallas Mavericks',
 'Houston Rockets',
 'New York Knicks',
 'Phoenix Suns',
 'Boston Celtics',
 'Sacramento Kings']


for team in team_list:
    with open(f'./app_assets/{team}.svg', 'r') as logo:
        team_logo = logo.read()

    team_logos[team] = team_logo


def create_feature_importance_plot(row):
    top_local_features = []
    top_local_values = []

    for feature, value in ast.literal_eval(row['shap_list']).items():
        top_local_features.append(feature)
        top_local_values.append(value)

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x= list(reversed(top_local_values[0:10])),
            y= list(reversed(top_local_features[0:10])),
            orientation='h'
        )
    )
    fig.update_layout(title_text=f'Local Feature Importance for {row["TEAM_NAME_AWAY"]} @ {row["TEAM_NAME_HOME"]}')

    return fig


def create_eda_1():
    fig = px.histogram(sample_data, x='home_cover_spread')
    fig.update_layout(title_text='Target Variable Distribution')
    return fig

def create_eda_2():
    fig = px.histogram(underdog_cover, x='Description')
    fig.update_layout(title_text='Distribution of Underdog and Favorite Point Spread Covers')
    return fig
