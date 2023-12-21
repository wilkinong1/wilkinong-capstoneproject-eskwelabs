import streamlit as st
import app_files as data
import app_functions as funcs
from datetime import datetime, timedelta
from streamlit_shap import st_shap
import pandas as pd

st.set_page_config(page_title='Capstone Project App', page_icon=None, layout="wide", initial_sidebar_state='collapsed')
st.set_option('deprecation.showPyplotGlobalUse', False)

if 'filtered_data' not in st.session_state:
    st.session_state['filtered_data'] = funcs.load_filtered(data.results, str(datetime.now().date() - timedelta(days=14)), str(datetime.now().date()), [])
if 'charts' not in st.session_state:
    st.session_state['charts'] = funcs.load_charts(data.results, str(datetime.now().date() - timedelta(days=14)), str(datetime.now().date()), [])
if 'charts_static' not in st.session_state:
    st.session_state['charts_static'] = funcs.load_charts_static(data.results_full)

upcoming_tab, current_results_tab, global_importance_tab, project_overview = st.tabs(['Upcoming', 'Results', 'Global Feature Importance', 'Project Overview'])

dates = sorted(list(set(st.session_state['filtered_data'][0]['GAME_DATE'])), reverse=True)

def main():
    with current_results_tab:
        st.title('2023-2024 Season Results')

        st.divider()

        st.header('Overall Model Results')
        
        date_range = f'{st.session_state["filtered_data"][1]} - {st.session_state["filtered_data"][2]}'
        correct_results = len(st.session_state['filtered_data'][0][st.session_state['filtered_data'][0]['home_cover_spread'] == st.session_state['filtered_data'][0]['predictions']])
        correct_results_full = len(data.results_full[data.results_full['home_cover_spread'] == data.results_full['predictions']])

        good_predictions_len = len(st.session_state['filtered_data'][0][st.session_state['filtered_data'][0]['pred_confidence'] == 'Good'])
        good_predictions_len_full = len(data.results_full[data.results_full['pred_confidence'] == 'Good'])

        correct_results_good = len(st.session_state['filtered_data'][0][(st.session_state['filtered_data'][0]['home_cover_spread'] == st.session_state['filtered_data'][0]['predictions']) & (st.session_state['filtered_data'][0]['pred_confidence'] == 'Good')])
        correct_results_good_full = len(data.results_full[(data.results_full['home_cover_spread'] == data.results_full['predictions']) & (data.results_full['pred_confidence'] == 'Good')])
        metric_col1, metric_col2 = st.columns(spec=[1, 1], gap='medium')
        with metric_col1.container():
            metric_subcol1, metric_subcol2 = st.columns(spec=[1, 1], gap='medium')

            metric_subcol1.metric(f'All Predictions Accuracy ({date_range})',f'{round((correct_results/len(st.session_state["filtered_data"][0])*100), 2)}%')
            metric_subcol1.metric(f'Good Confidence Accuracy ({date_range})',f'{round((correct_results_good/good_predictions_len*100), 2)}%')

            metric_subcol2.metric(f'All Predictions Accuracy (Entire Season)', f'{round((correct_results_full/len(data.results_full)*100), 2)}%')
            metric_subcol2.metric(f'Good Confidence Accuracy (Entire Season)',f'{round((correct_results_good_full/good_predictions_len_full*100), 2)}%')
        
        with metric_col2.container():
            chart_subcol1, chart_subcol2 = st.columns(spec=[1, 1], gap='small')

            chart_subcol1.plotly_chart(st.session_state['charts'], use_container_width=True)
            chart_subcol2.plotly_chart(st.session_state['charts_static'], use_container_width=True)

        st.divider()

        filter_col1, filter_col2, filter_col3 = st.columns(spec=[1,1,3], gap='small')
        date_select = filter_col1.date_input('Select Date Range', 
                                    value=(datetime.strptime(str(st.session_state['filtered_data'][0]["GAME_DATE"].min()), "%Y-%m-%d %H:%M:%S").date(), datetime.strptime(str(st.session_state['filtered_data'][0]["GAME_DATE"].max()), "%Y-%m-%d %H:%M:%S").date()),
                                    min_value=datetime.strptime(str(data.results_full["GAME_DATE"].min()), "%Y-%m-%d %H:%M:%S").date(), 
                                    max_value=datetime.strptime(str(data.results_full["GAME_DATE"].max()), "%Y-%m-%d %H:%M:%S").date()
                                )
        team_select = filter_col2.multiselect('Select Teams', data.team_list)
        def update_data():
            st.session_state['filtered_data'] = funcs.load_filtered(data.results, date_select[0], date_select[1], team_select)
            st.session_state['charts'] = funcs.load_charts(data.results, date_select[0], date_select[1], team_select)

        filter_col1.button("Filter", type='secondary', on_click=update_data)
        
        st.header('Daily Results')
        for date in dates:
            date_games = st.session_state['filtered_data'][0][st.session_state['filtered_data'][0]['GAME_DATE'] == date]
            
            st.subheader(datetime.strptime(str(date), "%Y-%m-%d %H:%M:%S").date())
            for _, row in date_games.iterrows():
                col1, col2 = st.columns(spec=[2, 2.5], gap='small')
                with col1.expander(f'{row["TEAM_NAME_AWAY"]} ({row["hcp_tipoff"]*-1}) @ {row["TEAM_NAME_HOME"]} ({row["hcp_tipoff"]})'):
                    sub_col_ex1, sub_col_ex2, sub_col_ex3, sub_col_ex4 = st.columns(spec=[1, 1, 1, 1], gap='small')

                    sub_col_ex2.image(data.team_logos[row['TEAM_NAME_AWAY']], use_column_width=True)
                    sub_col_ex2.markdown(f"<h4 style='text-align: center;'>{row['TEAM_NAME_AWAY']} ({row['hcp_tipoff']*-1})</h4>", unsafe_allow_html=True)

                    sub_col_ex3.image(data.team_logos[row['TEAM_NAME_HOME']], use_column_width=True)
                    sub_col_ex3.markdown(f"<h4 style='text-align: center;'>{row['TEAM_NAME_HOME']} ({row['hcp_tipoff']})</h4>", unsafe_allow_html=True)

                    sub_col_ex1.markdown("")
                    sub_col_ex1.markdown("")
                    sub_col_ex1.markdown("")
                    sub_col_ex1.markdown("")
                    sub_col_ex1.markdown("")
                    sub_col_ex1.markdown(f"<h1 style='text-align: center;'>{row['PTS_AWAY']}</h1>", unsafe_allow_html=True)
                    sub_col_ex4.markdown("")
                    sub_col_ex4.markdown("")
                    sub_col_ex4.markdown("")
                    sub_col_ex4.markdown("")
                    sub_col_ex4.markdown("")
                    sub_col_ex4.markdown(f"<h1 style='text-align: center;'>{row['PTS_HOME']}</h1>", unsafe_allow_html=True)
                    
                    if row['hcp_tipoff'] > 0:
                        home_favorite = False
                        underdog = row['TEAM_NAME_HOME']
                    else:
                        home_favorite = True
                        underdog = row['TEAM_NAME_AWAY']

                    if (home_favorite) and row['home_cover_spread'] == 0:
                        excite = True
                        excite_emoji = ':point_up_2:'
                    elif (home_favorite) and row['home_cover_spread'] == 1:
                        excite = False
                        excite_emoji = ':point_down:'
                    elif not (home_favorite) and row['home_cover_spread'] == 1:
                        excite = True
                        excite_emoji = ':point_up_2:'
                    else:
                        excite = False
                        excite_emoji = ':point_down:'

                    if excite:
                        actual = f'{excite_emoji} {underdog} covered the spread, this game was more exciting than expected!'
                    else:
                        actual = f'{excite_emoji} {underdog} was not able to cover the spread!'
                    
                    if row['home_cover_spread'] == row['predictions']:
                        prediction = 'The model predicted this correctly'
                        pred_emoji = ':white_check_mark:'
                    else:
                        prediction = 'The model was not able to predict this correctly'
                        pred_emoji = ':x:'

                    st.markdown(f'')
                    st.markdown(f'')
                    st.markdown(f'**Results and Model Summary**')
                    st.markdown(f'{actual}')
                    st.markdown(f'{pred_emoji} {prediction}')

                    if row["pred_confidence"] == 'Good':
                        pred_emoji = ':large_green_circle:'
                    else:
                        pred_emoji = ':large_orange_circle:'

                    st.markdown(f'{pred_emoji} Model prediction confidence: {row["pred_confidence"]}')

                with col2.expander('Feature Importance'):
                    shap_r_subcol1,shap_r_subcol2, shap_r_cubcol3 = st.columns(spec=[1, 5, 1], gap='small')
                    shap_r_subcol2.plotly_chart(data.create_feature_importance_plot(row))

    with upcoming_tab:
        st.title("Upcoming Games")
        st.divider()
        st.header(list(set(data.upcoming['GAME_DATE']))[0])
        for _, row in data.upcoming.iterrows():
            col1_upcoming, col2_upcoming = st.columns(spec=[2, 2.5], gap='small')
            with col1_upcoming.expander(f'{row["TEAM_NAME_AWAY"]} ({row["hcp_tipoff"]*-1}) @ {row["TEAM_NAME_HOME"]} ({row["hcp_tipoff"]})'):
                sub_col_up1, sub_col_up2, sub_col_up3, sub_col_up4 = st.columns(spec=[1, 1, 1, 1], gap='small')

                sub_col_up2.image(data.team_logos[row['TEAM_NAME_AWAY']], use_column_width=True)
                sub_col_up2.markdown(f"<h4 style='text-align: center;'>{row['TEAM_NAME_AWAY']} ({row['hcp_tipoff']*-1})</h4>", unsafe_allow_html=True)
                
                sub_col_up3.image(data.team_logos[row['TEAM_NAME_HOME']], use_column_width=True)
                sub_col_up3.markdown(f"<h4 style='text-align: center;'>{row['TEAM_NAME_HOME']} ({row['hcp_tipoff']})</h4>", unsafe_allow_html=True)
                
                st.markdown(f'')
                st.markdown(f'')
                st.markdown(f'**Model Predictions and Confidence**')
                
                if row['hcp_tipoff'] > 0:
                    home_favorite = False
                    underdog = row['TEAM_NAME_HOME']
                else:
                    home_favorite = True
                    underdog = row['TEAM_NAME_AWAY']
                
                if row['MODEL_PREDICTION'] == 1 and home_favorite:
                    statement = f'The model predicts that {underdog} WILL NOT cover the point spread!'
                    statement_emoji = ':point_down:'
                elif row['MODEL_PREDICTION'] == 0 and home_favorite:
                    statement = f'The model predicts that {underdog} WILL cover the point spread!'
                    statement_emoji = ':point_up_2:'
                elif row['MODEL_PREDICTION'] == 1 and not (home_favorite):
                    statement = f'The model predicts that {underdog} WILL cover the point spread!'
                    statement_emoji = ':point_up_2:'
                else:
                    statement = f'The model predicts that {underdog} WILL NOT cover the point spread!'
                    statement_emoji = ':point_down:'
                
                if row["pred_confidence"] == 'Good':
                    pred_emoji = ':large_green_circle:'
                else:
                    pred_emoji = ':large_orange_circle:'
                st.write(f'{statement_emoji} {statement}')
                st.write(f'{pred_emoji} Model prediction confidence: {row["pred_confidence"]}')

            with col2_upcoming.expander('Feature Importance'):
                shap_subcol1,shap_subcol2, shap_cubcol3 = st.columns(spec=[1, 5, 1], gap='small')
                shap_subcol2.plotly_chart(data.create_feature_importance_plot(row))

    with global_importance_tab:
        st.title("Model Global Feature Importance")
        st.divider()
        gl_col1, gl_col2 = st.columns(spec=[1,1], gap='medium')
        with gl_col1.container():
            st_shap(funcs.show_global_importance(), height=1000, width=800)
        
        with gl_col2.container():
            st.markdown('''
                        # Feature Importance
                        - To the left is a SHAP beeswarm plot that summarizes the top 20 most important features that the model uses to make its prediction.
                        - On the Upcoming and Results tabs, you are able to see the local feature importance for each of the predictions made by the model.
                        - When diving into explainability. It makes sense that the point spread has the highest impact on model output since it is what is being compared against.
                        - Model explainability and seeing which features nudged the model towards making its prediction is important to go beyond the blackbox of some models. It can tie it with a person’s own domain knowledge and expectations to help the user ultimately decide whether or not the choice is right.
                        ## Glossary:
                        - hcp_tipoff: The point spread for the home team. If the point spread is a positive value, the home team is the underdog, if it is a negative value, the home team is favorite to win.
                        - AST_PCT_HOME: Percentage of field goals by the home team that are assisted.
                        - W_PCT_L10_HOME/AWAY: The win percentage of the home/away team from its last 10 games.
                        - OREB_PCT_HOME: The percentage of available offensive rebounds the home team obtains.
                        - TM_TOV_PCT_L10_AWAY: The percentage of plays that end in a turnover for the away team's last 10 games.
                        - TS_PCT_L10_HOME: The true shooting percentage for the home team for it's last 10 games. True shooting percentage is: Points/ [2*(Field Goals Attempted+0.44*Free Throws Attempted)]
                        - DAYS_REST_HOME/AWAY: Numbers of days rest for the home/away team before coming into the game.
                        - REB_PCT_HOME/AWAY: The percentage of available defensive rebounds a team obtained.
                        - DEF_RATING_DIFF: The difference between the teams' defensive rating. The defensive rating is the number of points allowed per 100 posessions by a team.
                        - DREB_PCT_L10_AWAY: The percentage of available defensive rebounds obtained by the away team in its last 10 games.
                        - OFF_RATING_DIFFERENCE: The difference between the teams' offensive rating. The offensive rating is the number of points scored per 100 posessions by a team.
                        - E_PACE_L10_DIFF: The difference between the teams' pace. Pace is the number of posessions per 48 minutes for the team.
                        ''')
    
    with project_overview:
        st.title('League Pass Companion: Elevating the NBA Fan Experience')
        st.divider()
        st.header(':basketball: Introduction')
        st.markdown(''' 
                - The NBA is a huge business. It is financially the biggest basketball league globally.
                - Last season it generated \$10 billion in revenue with \$3 billion in profit, but viewership is well below it peak from the late 90s. (https://huddleup.substack.com/p/how-the-nba-became-a-10-billion-annual)
                - The rise of streaming and the general decline in television viewership has contributed to the decline
                - It is important for the League to showcase its best product to continue grow its fan base and revenue as TV broadcasters downsize the number of games they televise. (https://www.forbes.com/sites/bradadgate/2023/10/23/how-the-next-nba-media-rights-negotiations-will-be-different/)
                - To continue to grow its fan base and revenue, the NBA must showcase its best product.
                - Fans, TV Broadcasters have limited resource and time.
                    ''')
        st.divider()
        st.header(':basketball: Entertainment Value from NBA Games')
        st.markdown('''
                - Fans have limited time and resource, **it's important for fans to maximize the entertainment value from NBA games**, which is key to fan experience
                - This project aims to improve fan experience by maximizing entertainment value over the long term by predicting games that will be more entertaining than expected.
                    ''')
        st.divider()
        st.header(':basketball: The Point Spread')
        st.markdown('''
                - The point spread will serve as the basis for expectations in terms of team performance
                - When the underdog is able to cover the point spread, the game is classified as 'more exciting' as the underdog is able overperform against expectations, leading to closer games than expected, which in theory should be more exciting games.
                - Here's a good article on how the point spread works: https://www.forbes.com/betting/sports-betting/what-does-the-point-spread-mean/
                    ''')
        st.divider()
        st.header(':basketball: Data Gathering and Description')
        st.markdown('''
                - External Data Sources include the following:
                    1. NBA Python Package for NBA Statistics: (https://github.com/swar/nba_api)
                    2. BetsAPI for historical point spread information: (https://betsapi.com/docs/)
                    3. Web scraping https://www.sportsbookreview.com/ for missing point spreads and https://projects.fivethirtyeight.com/complete-history-of-the-nba/#nuggets for team Elo. (https://fivethirtyeight.com/features/how-we-calculate-nba-elo-ratings/ for Elo explanation)
                - Dataset shape is 6969 x 73 columns after aggregation, preprocessing, and cleaning.
                - Below is a sample row from the data set. It contains team statistics, added features such as days rest, team Elo, and match information like date, season, the point spread, and the target variable.
                    ''')
        st.dataframe(data=data.sample_data.head(1))
        st.divider()
        st.header(':basketball: EDA')
        st.markdown('''- The dataset is balanced with a distribution of 49.3\% and 50.7\% for the target variable''')
        st.plotly_chart(data.create_eda_1())
        st.markdown('''- Home court advantage is taken into account as seen in the spread information for the home team:''')
        st.dataframe(pd.DataFrame({'Average Home Team Spread': [-2.23], 'Average Home Team Spread (Home Favored)': [-6.49], 'Average Home Team Spread (Home Underdog)': [5.06], 'Max Spread': [17.0], 'Min Spread': [-19.5]}))
        st.markdown('''- Going back to our problem, the underdog covers the spread 50\% of the time, this means that picking whether or not the underdog will overperform against expectations is a toss up. Going above this will improve how much entertainment value you can derive from NBA games in the long run.''')
        st.plotly_chart(data.create_eda_2())
        st.divider()
        st.header(':basketball: Model Pipeline')
        with st.container():
            st.image('./diagram2.drawio.png')
        st.divider()
        st.header(':basketball: Model Baselining')
        st.markdown('''
                    - The evaluation metric used is Accuracy. This was considered becase dataset is balanced and the target variable is binary.
                    - Target accuracy is 52.4\% since we are predicting against the point spread, which is generally used in sports betting. It makes sense to set the target as the win rate you would need in order to make a profit given that the bookmaker takes a 4.5% margin for every bet.
                    - There has not be a lot of publicly available studies that predict against the spread, the best that I could find only got around 52.1\% accuracy. (https://github.com/NBA-Betting/NBA_Betting)
                    - Various base models in combination with different scalers were used as baseline models. Tree based ensemble models were pre-pruned to prevent highly overfitted models
                    - The baseline results for models that will go on to hyperparameter tuning are shown below:
                ''')
        st.dataframe({
            'Model': ['LogisticRegressor', 'GradientBoostingClassifier', 'SGDClassifier', 'LinearSVC', 'AdaBoostClassifier'],
            'Scaler': ['MinMaxScaler', 'StandardScaler', 'RobustScaler', 'StandardScaler', 'RobustScaler'],
            'Train Result': [0.54, 0.63, 0.52, 0.53, 0.59],
            'Validation Result': [0.53, 0.54, 0.52, 0.52, 0.53]})
        st.divider()
        st.header(':basketball: Results')
        st.markdown('''
            - After hyperparameter tuning, the best performing model against the holdout was a Logistic Regression model with the following parameters:
        ''')
        st.image('./model_pipe.png')
        st.markdown('''
            - Unfortunately, model performance against the holdout was only at around 51% which falls short of the target accuracy.
            - To maximize the model, the .predict_proba method of the classifier was used. When using the model to only predict an outcome when its confidence is above 0.524 gave much better results at 54\% over 556 predictions.
            - in the same way that a sports bettor cannot bet on every game, fans also can't consider watching all games due to limited time and resource. In this vein, the model should only be used when confidence is high so that fans can maximize the limited time and resource they have wathcing basketball games.
            - A 54% hit rate, long term will derive positive entertainment value versus expectations.
            - As seen in the results tab, the model has been doing fairly well predicting whether or not the underdog will cover the spread. Including only good quality predictions also results in an accuracy greater than the target.
        ''')
        st.divider()
        st.header(':basketball: Explainability and Application')
        st.markdown(''' 
            - Check Global Feature Importance tab for more on explainability.
            - This Streamlit App is essentially a practical application of the model for this project.
            - If you haven't already noticed, this project is actually a thinly veiled sports betting project. 'Entertainment value' can ba substituted almost 1:1 with 'profit'
            - It does make sense that sports betting and entertainment derived from games are closely related. It's always more fun to watch games when there are stakes involved.
            - What betting strategy you choose to employ using the predictions the model gives is entirely up to you. 
        ''')
        st.divider()
        st.header(':basketball: Limitations and Future Iterations')
        st.markdown('''
            - Data only covers team statistics and performance and does not go into individual player performance
            - Scores could be improved when using a different way of measuring 'entertainment value'
            - Trimming features based on the Logistic Regression .coef_ attribute and retraining the model
            - Continuously adding the current season data into the dataset for predicting upcoming games
        ''')



if __name__ == "__main__":
    main()
