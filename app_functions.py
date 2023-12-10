import pandas as pd
from datetime import datetime, timedelta
import shap
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import pickle
import matplotlib.pyplot as plt
import plotly.express as px

def load_filtered(df, start, end, teams):
    if len(teams) > 0:
        df = df[(df['TEAM_NAME_HOME'].isin(teams)) | (df['TEAM_NAME_AWAY'].isin(teams))]

    df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'])
    df['date_index'] = df['GAME_DATE']
    df.set_index('date_index', inplace=True)
    df.sort_index(ascending=True, inplace=True)
    df = df[start:end]
    min_date = datetime.strptime(str(df["GAME_DATE"].min()), "%Y-%m-%d %H:%M:%S").date()
    max_date = datetime.strptime(str(df["GAME_DATE"].max()), "%Y-%m-%d %H:%M:%S").date()
    return (df, min_date, max_date)

def load_charts(df, start, end, teams):
    if len(teams) > 0:
        df = df[(df['TEAM_NAME_HOME'].isin(teams)) | (df['TEAM_NAME_AWAY'].isin(teams))]
    df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'])
    df['date_index'] = df['GAME_DATE']
    df.set_index('date_index', inplace=True)
    df.sort_index(ascending=True, inplace=True)
    df = df[start:end]
    df['correct'] = df.apply(lambda x: 1 if x['home_cover_spread'] == x['predictions'] else 0, axis=1)
    for_plot = df.groupby('GAME_DATE').agg({'correct':'sum', 'predictions': 'count'})
    for_plot['acc'] = for_plot['correct']/for_plot['predictions']
    fig = px.line(for_plot.reset_index(), x='GAME_DATE', y='acc')
    fig.add_hline(y=0.524, line_dash='dash', line_color='white')
    fig.update_layout(yaxis_title='Accuracy', title_text='Daily Accuracy (Filtered)')
    return fig

def load_charts_static(df):
    df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'])
    df['date_index'] = df['GAME_DATE']
    df.set_index('date_index', inplace=True)
    df.sort_index(ascending=True, inplace=True)
    df['correct'] = df.apply(lambda x: 1 if x['home_cover_spread'] == x['predictions'] else 0, axis=1)
    for_plot = df.groupby('GAME_DATE').agg({'correct':'sum', 'predictions': 'count'})
    for_plot['acc'] = for_plot['correct']/for_plot['predictions']
    fig = px.line(for_plot.reset_index(), x='GAME_DATE', y='acc')
    fig.add_hline(y=0.524, line_dash='dash', line_color='white')
    fig.update_layout(yaxis_title='Accuracy', title_text='Daily Accuracy (All)')
    return fig




def show_global_importance():
    data_train = pd.read_csv('./train_test_data.csv')

    def dataset_filters(df, min_games_played, seasons=None):
        df = df[(df['GP_HOME'] >= min_games_played) & (df['GP_AWAY'] >= min_games_played)]
        if seasons:
            df = df[df['SEASON_YEAR'].isin(seasons)]
        return df

    dataset_filters(data_train, 10)

    data_train.drop(columns=['SEASON_YEAR', 'GAME_DATE', 'GP_HOME', 'GP_AWAY'], inplace=True)
    X_ = data_train.drop(['home_cover_spread'], axis=1)
    y_ = data_train['home_cover_spread']
    (X_trainval, X_holdout, y_trainval, y_holdout) = train_test_split(X_, y_, random_state=24, test_size=0.25, stratify=y_)

    
    model = pickle.load(open('log_reg.pkl', 'rb'))
    explainer = shap.Explainer(model.named_steps['classifier'], MinMaxScaler().fit_transform(X_trainval), feature_names=X_trainval.columns)
    x_shap = MinMaxScaler().fit_transform(X_holdout)
    shap_values = explainer.shap_values(x_shap)
    return shap.summary_plot(shap_values, MinMaxScaler().fit_transform(X_holdout), feature_names=X_trainval.columns, max_display=15)