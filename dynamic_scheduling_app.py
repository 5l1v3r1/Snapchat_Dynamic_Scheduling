import streamlit as st
import math
import pandas as pd
import numpy as np
from numpy.ma.core import log

from neuralprophet import NeuralProphet
import chart_studio.plotly as py
from plotly import graph_objs as go
from neuralprophet.benchmark import Dataset, NeuralProphetModel, SimpleExperiment, CrossValidationExperiment

from google.oauth2 import service_account
from google.cloud import bigquery

#import streamlit_google_oauth as oauth
#import os
#from google.cloud import secretmanager

#Page Configuration 
st.set_page_config(
                    page_title='Snapchat Dynamic Scheduling', 
                    page_icon = 'https://w7.pngwing.com/pngs/481/484/png-transparent-snapchat-logo-snap-inc-social-media-computer-icons-snapchat-text-logo-smiley.png', 
                    layout='wide'
                  )
# header of the page 
html_temp = """ 
            <div style ="background-color:#00008B; border: 8px darkblue; padding: 18px; text-align: right">
            <!<img src="https://www.rewindandcapture.com/wp-content/uploads/2014/04/snapchat-logo.png" width="100"/>>
            <h1 style ="color:lightgrey;text-align:center;">Snapchat Dynamic Scheduling</h1>
            </div> 
            """
st.markdown(html_temp,unsafe_allow_html=True) 

#Minor template configurations 
css_background = """
                  <style>
                  h1   {color: darkblue;}
                  p    {color: darkred;}
                  </style>
                  """
st.markdown(css_background,unsafe_allow_html=True) 

# Create API client.
credentials = service_account.Credentials.from_service_account_info(st.secrets["gcp_service_account"])
#client = bigquery.Client(credentials=credentials)

#Ignore warning
st.set_option('deprecation.showPyplotGlobalUse', False)

#Creating Functions 

def forecast_totalview(choose_episode, choose_hours):
    #Load in episode
    data = df[df['story_id'].isin([choose_episode])]
    data = data.loc[:, ['interval_time', 'topsnap_views']]
    data = data.rename(columns = {'interval_time': 'ds', 'topsnap_views':'y'})
    data = data.drop_duplicates(subset='ds')
    data = data.astype({'y' : 'int32'})

    # Train and load model
    m = NeuralProphet(num_hidden_layers=2,
                    d_hidden=4,
                    seasonality_mode='muplicative',
                    learning_rate=5.0,
                    batch_size=40,
                    loss_func='mse'
                    )
  
    metrics = m.fit(data, freq='H')
  
    future = m.make_future_dataframe(data, periods=choose_hours, n_historic_predictions=len(data)) 
    prediction = m.predict(future)

    #Get Confidence Interval upper/lower bounds (0.95)
    bounds = prediction['yhat1']
    average_data = []
    for ind in range(len(bounds)):
      average_data.append(np.mean(bounds[0:ind+1]))

    prediction['running_mean'] = average_data

    std_data = []
    for ind in range(len(bounds)):
      std_data.append(np.std(bounds[0:ind+1]))

    prediction['running_std'] = std_data

    prediction['n'] = prediction.index.to_list()
    prediction['n'] = prediction['n'] + 1

    #95% confience interval with zscore of 1.96
    prediction['ci'] = 1.96 * prediction['running_std'] / np.sqrt(prediction['n'])
    prediction['yhat_lower'] = prediction['yhat1'] - prediction['ci']
    prediction['yhat_upper'] = prediction['yhat1'] + prediction['ci']

    #Visualize Model 
    yhat = go.Scatter(x = prediction['ds'], 
                    y = prediction['yhat1'],
                    mode = 'lines',
                    marker = {'color': 'blue'},
                    line = {'width': 4},
                    name = 'Forecast',
                    )
    yhat_lower = go.Scatter(x = prediction['ds'],
                          y = prediction['yhat_lower'],
                          marker = {'color': 'powderblue'},
                          showlegend = False,
                          )
    yhat_upper = go.Scatter(x = prediction['ds'],
                          y = prediction['yhat_upper'],
                          fill='tonexty',
                          fillcolor = 'powderblue',
                          name = 'Confidence (95%)',
                          mode = 'none'
                          )
    actual = go.Scatter(x = data['ds'],
                      y = data['y'],
                      mode = 'markers',
                      marker = {'color': '#fffaef','size': 10,'line': {'color': '#000000',
                                                                      'width': 0.8}},
                      name = 'Actual'
                      )
  
    layout = go.Layout(yaxis = {'title': 'Topsnaps',},
                     hovermode = 'x',
                     xaxis = {'title': 'Hours/Days'},
                     margin = {'t': 20,'b': 50,'l': 60,'r': 10},
                     legend = {'bgcolor': 'rgba(0,0,0,0)'})
  
    data = [yhat_lower, yhat_upper, yhat, actual]

    #Get Episode name
    episode_df = df[df['story_id'].isin([choose_episode])]
    episode_name = episode_df.head(1)['title'].values[0]

    #Get Channel name 
    channel_df = benchmarks[benchmarks['name'].isin(episode_df.name)]
    channel_name = channel_df.head(1)['name'].values[0]

    #Get values to visualize predicted performance in the title 
    start2 = future.dropna().tail(1)['y'].values[0]
    end2 = prediction.tail(1)['yhat1'].values[0]
    number = round(end2-start2)

    start_end = prediction.tail(24)
    start = start_end.head(1)['yhat1'].values[0]
    end = start_end.tail(1)['yhat1'].values[0]
    last_24 = round(end-start)

    #Get benchmarks
    def get_benchmarks(choose):
      b_channel = benchmarks[benchmarks['name'].isin(episode_df.name)]
      b_channel = b_channel.loc[b_channel['ranking'] == choose, ['topsnap_views_total']]
      channel_bench = b_channel['topsnap_views_total'].mean()
      
      return channel_bench

    if choose_hours <= 24:
      channel_bench = get_benchmarks(24)
      day = 'Day 1'
      last_24 = end

    elif ((choose_hours > 24) and (choose_hours <= 48)):
      channel_bench = get_benchmarks(48)
      day = 'Day 2'

    elif ((choose_hours > 48) and (choose_hours <= 72)):
      channel_bench = get_benchmarks(72)
      day = 'Day 3'

    elif ((choose_hours > 72) and (choose_hours <= 96)):
      channel_bench = get_benchmarks(96)
      day = 'Day 4'

    elif ((choose_hours > 96) and (choose_hours <= 120)):
      channel_bench = get_benchmarks(120)
      day = 'Day 5'

    elif ((choose_hours > 120) and (choose_hours <= 144)):
      channel_bench = get_benchmarks(144)
      day = 'Day 6'

    elif ((choose_hours > 144) and (choose_hours <= 168)):
      channel_bench = get_benchmarks(168)
      day = 'Day 7'

    fig = go.Figure(data= data, layout=layout)
    
    fig.update_layout(title={'text': (f'<b>{episode_name} - {channel_name}</b><br><br><sup>Total Topsnap Prediction = <b>{round(end):,}</b><br>{day} Topsnap Prediction = <b>{last_24:,}<b></sup>'),
                           'y':0.91,
                           'x':0.075,
                           'font_size':22})

    fig.add_hline(y=channel_bench, line_dash="dot", line_color='purple',
                annotation_text=(f"Channel Avg at {choose_hours}hrs: <b>{round(channel_bench):,}</b>"), 
                annotation_position="bottom right", annotation_font_size=14,
                annotation_font_color="purple")

    return fig

def forecast_dailyview(choose_episode, choose_hours):
    #Load in episode
    data = df[df['story_id'].isin([choose_episode])]
    data = data.loc[:, ['interval_time', 'topsnap_views']]
    data = data.rename(columns = {'interval_time': 'ds', 'topsnap_views':'y'})
    data = data.drop_duplicates(subset='ds')
    data = data.astype({'y' : 'int32'})
    hours_number = choose_hours - len(data)

    # Train and load model
    m = NeuralProphet(num_hidden_layers=2,
                    d_hidden=4,
                    seasonality_mode='muplicative',
                    learning_rate=5.0,
                    batch_size=40,
                    loss_func='mse'
                    )
  
    metrics = m.fit(data, freq='H')
  
    future = m.make_future_dataframe(data, periods=choose_hours, n_historic_predictions=len(data)) 
    prediction = m.predict(future)

    #Daily dataframe
    show_prediction = prediction.iloc[-24:]
    show_prediction['y_daily'] = ((show_prediction.loc[:, ['y']]) - (show_prediction.loc[:, ['y']].shift(+1))).cumsum()
    show_prediction['yhat_daily'] = ((show_prediction.loc[:, ['yhat1']]) - (show_prediction.loc[:, ['yhat1']].shift(+1))).cumsum()

    #Get Confidence Interval upper/lower bounds (0.95)
    bounds = show_prediction['yhat_daily']
    average_data = []
    for ind in range(len(bounds)):
      average_data.append(np.mean(bounds[0:ind+1]))
    show_prediction['running_mean'] = average_data

    std_data = []
    for ind in range(len(bounds)):
      std_data.append(np.std(bounds[0:ind+1]))
    show_prediction['running_std'] = std_data

    show_prediction = show_prediction.reset_index().drop(columns=['index'])
    show_prediction['n'] = show_prediction.index.to_list()
    show_prediction['n'] = show_prediction['n'] + 1

    #95% confience interval with zscore of 1.96
    show_prediction['ci'] = 1.96 * show_prediction['running_std'] / np.sqrt(show_prediction['n'])
    show_prediction['yhat_lower'] = show_prediction['yhat_daily'] - show_prediction['ci']
    show_prediction['yhat_upper'] = show_prediction['yhat_daily'] + show_prediction['ci']

    #Visualize Model 
    yhat = go.Scatter(x = show_prediction['ds'], 
                    y = show_prediction['yhat_daily'],
                    mode = 'lines',
                    marker = {'color': 'blue'},
                    line = {'width': 4},
                    name = 'Forecast',
                    )
    yhat_lower = go.Scatter(x = show_prediction['ds'],
                          y = show_prediction['yhat_lower'],
                          marker = {'color': 'powderblue'},
                          showlegend = False,
                          )
    yhat_upper = go.Scatter(x = prediction['ds'],
                          y = show_prediction['yhat_upper'],
                          fill='tonexty',
                          fillcolor = 'powderblue',
                          name = 'Confidence (95%)',
                          mode = 'none'
                          )
    actual = go.Scatter(x = data['ds'],
                      y = show_prediction['y_daily'],
                      mode = 'markers',
                      marker = {'color': '#fffaef','size': 10,'line': {'color': '#000000',
                                                                      'width': 0.8}},
                      name = 'Actual'
                      )
  
    layout = go.Layout(yaxis = {'title': 'Topsnaps',},
                     hovermode = 'x',
                     xaxis = {'title': 'Hours/Days'},
                     margin = {'t': 20,'b': 50,'l': 60,'r': 10},
                     legend = {'bgcolor': 'rgba(0,0,0,0)'})
  
    data = [yhat_lower, yhat_upper, yhat, actual]

    #Get Episode name
    episode_df = df[df['story_id'].isin([choose_episode])]
    episode_name = episode_df.head(1)['title'].values[0]

    #Get Channel name 
    channel_df = benchmarks[benchmarks['name'].isin(episode_df.name)]
    channel_name = channel_df.head(1)['name'].values[0]

    #Get values to visualize predicted performance in the title 
    start2 = future.dropna().tail(1)['y'].values[0]
    end2 = prediction.tail(1)['yhat1'].values[0]
    number = round(end2-start2)

    start_end = prediction.tail(24)
    start = start_end.head(1)['yhat1'].values[0]
    end = start_end.tail(1)['yhat1'].values[0]
    last_24 = round(end-start)

    #Get benchmarks
    def get_benchmarks(choose):
      b_channel = benchmarks[benchmarks['name'].isin(episode_df.name)]
      b_channel = b_channel.loc[b_channel['ranking'] == choose, ['topsnap_daily_diff']]
      channel_bench = b_channel['topsnap_daily_diff'].mean()
      return channel_bench

    if choose_hours <= 24:
      b_channel = benchmarks[benchmarks['name'].isin(episode_df.name)]
      b_channel = b_channel.loc[b_channel['ranking'] == 24, ['topsnap_views_total']]
      channel_bench = b_channel['topsnap_views_total'].mean()
      day = 'Day 1'
      last_24 = end

    elif ((choose_hours > 24) and (choose_hours <= 48)):
      channel_bench = get_benchmarks(48)
      day = 'Day 2'

    elif ((choose_hours > 48) and (choose_hours <= 72)):
      channel_bench = get_benchmarks(72)
      day = 'Day 3'

    elif ((choose_hours > 72) and (choose_hours <= 96)):
      channel_bench = get_benchmarks(96)
      day = 'Day 4'

    elif ((choose_hours > 96) and (choose_hours <= 120)):
      channel_bench = get_benchmarks(120)
      day = 'Day 5'

    elif ((choose_hours > 120) and (choose_hours <= 144)):
      channel_bench = get_benchmarks(144)
      day = 'Day 6'

    elif ((choose_hours > 144) and (choose_hours <= 168)):
      channel_bench = get_benchmarks(168)
      day = 'Day 7'

    fig = go.Figure(data= data, layout=layout)
    
    fig.update_layout(title={'text': (f'<b>{day} : {episode_name} - {channel_name}</b><br><br><sup>{day} Topsnap Prediction = <b>{last_24:,}</b><br>{hours_number:,}hr Topsnap Prediction = <b>{number:,}</b></sup>'),
                           'y':0.91,
                           'x':0.075,
                           'font_size':22})

    fig.add_hline(y=channel_bench, line_dash="dot", line_color='purple',
                annotation_text=(f"Channel Avg at {choose_hours}hrs: <b>{round(channel_bench):,}</b>"), 
                annotation_position="bottom right", annotation_font_size=14,
                annotation_font_color="purple")

    return fig

def tts_model():
    #Train and Test the  model
    m = NeuralProphet(num_hidden_layers=2,
                    d_hidden=4,
                    seasonality_mode='muplicative', 
                    learning_rate=5.0,
                    batch_size=40,
                    loss_func='mse'
                    )
    return m

def plot_loss(tts_episode):
    #Load in Episode
    data = df[df['story_id'].isin([tts_episode])]
    data = data.loc[:, ['interval_time', 'topsnap_views']]
    data = data.rename(columns = {'interval_time': 'ds', 'topsnap_views':'y'})
    data = data.drop_duplicates(subset='ds')
    data = data.astype({'y' : 'int32'})

    df_train, df_test = model.split_df(df=data, freq="H", valid_p=0.2)

    metrics_train = model.fit(df=df_train, freq="H", validation_df=df_test)
    metrics_test = model.test(df=df_test)
    
    plot_metrics = metrics_train[['MSELoss', 'MSELoss_val']]
    plot_metrics = plot_metrics.rename(columns={'MSELoss_val': 'Test', 'MSELoss':'Train'})
    plot_metrics = np.log(plot_metrics)

    return plot_metrics

def test_metrics(tts_episode):
    #Load in Episode
    data = df[df['story_id'].isin([tts_episode])]
    data = data.loc[:, ['interval_time', 'topsnap_views']]
    data = data.rename(columns = {'interval_time': 'ds', 'topsnap_views':'y'})
    data = data.drop_duplicates(subset='ds')
    data = data.astype({'y' : 'int32'})

    df_train, df_test = model.split_df(df=data, freq="H", valid_p=0.2)

    metrics_train = model.fit(df=df_train, freq="H", validation_df=df_test)
    metrics_test = model.test(df=df_test)

    return metrics_test

def crossvalidate_three(tts_episode):
  data = df[df['story_id'].isin([tts_episode])]
  new_data = data.loc[:, ['interval_time', 'topsnap_views']]
  new_data = new_data.rename(columns = {'interval_time': 'ds', 'topsnap_views':'y'})
  new_data = new_data.drop_duplicates(subset='ds')
  new_data = new_data.astype({'y' : 'int32'})

  episode_name = data.head(1)['title'].values[0]
  test = Dataset(df=new_data, name=episode_name, freq='MS')

  params = {"num_hidden_layers": 2,
          "d_hidden":4,
          "seasonality_mode":'muplicative',
          "learning_rate":5.0,
          "batch_size":40,
          "loss_func":'mse'}
          
  exp_cv = CrossValidationExperiment(model_class=NeuralProphetModel,
                                     params=params,
                                     data=test,
                                     metrics=["MASE", "MAE", "RMSE"],
                                     test_percentage=20,
                                     num_folds=3,
                                     fold_overlap_pct=0)
  
  result_train, result_test = exp_cv.run()
  dataframe = pd.DataFrame(result_test)
  
  return dataframe

def crossvalidate_five(tts_episode):
  data = df[df['story_id'].isin([tts_episode])]
  new_data = data.loc[:, ['interval_time', 'topsnap_views']]
  new_data = new_data.rename(columns = {'interval_time': 'ds', 'topsnap_views':'y'})
  new_data = new_data.drop_duplicates(subset='ds')
  new_data = new_data.astype({'y' : 'int32'})

  episode_name = data.head(1)['title'].values[0]
  test = Dataset(df=new_data, name=episode_name, freq='MS')

  params = {"num_hidden_layers": 2,
          "d_hidden":4,
          "seasonality_mode":'muplicative',
          "learning_rate":5.0,
          "batch_size":40,
          "loss_func":'mse'}
          
  exp_cv = CrossValidationExperiment(model_class=NeuralProphetModel,
                                     params=params,
                                     data=test,
                                     metrics=["MASE", "MAE", "RMSE"],
                                     test_percentage=15,
                                     num_folds=5,
                                     fold_overlap_pct=0)
  
  result_train, result_test = exp_cv.run()
  dataframe = pd.DataFrame(result_test)
  
  return dataframe

# Uses st.experimental_memo to only rerun when the query changes or after 30 min.
#@st.experimental_memo(ttl=1800)
@st.cache(ttl=1800)
def update_data():
    sql_query = ('''WITH cte AS (SELECT
                  non_fin.*,
                  dense_rank() over(partition by non_fin.story_id order by interval_time) ranking
    
                  FROM
                  ( --- non-financial numbers that are not aggregated
                        SELECT
                        topsnap_views, 
                        name, 
                        title,
                        datetime(published_at,"America/Toronto") published_at,
                        datetime(interval_time,"America/Toronto") interval_time,
                        story_id,
                        --metrics
                        avg_time_spent_per_user avg_time_spent_per_user,
                        completion_rate completion_rate,
                        screenshots,
                        shares,
                        subscribers,
                        total_time_viewed,
                        total_views,
                        unique_completers,
                        unique_topsnap_views,
                        unique_topsnaps_per_user,
                        unique_viewers,
                        drop_off_rate
    
                        FROM `distribution-engine.post_time_series.snap_post_metrics_30_minutes_with_diff` hourly
                        LEFT JOIN  EXTERNAL_QUERY(
                                                  "projects/distribution-engine/locations/us/connections/postgres",
                                                  """
                                                  SELECT story_id::TEXT,
                                                        published_at,
                                                        title,
                                                        name
                                                  FROM snap_studio_story_metric
                                                  LEFT JOIN snap_publishers USING (publisher_id)
                                                  ---where name in ('Crafty')
                                                     order by published_at desc
                                                  """
                                                  ) AS pub USING (story_id) 

                          LEFT JOIN  EXTERNAL_QUERY(
                                                    "projects/distribution-engine/locations/us/connections/postgres",
                                                    """
                                                    select
                                                        story_id::TEXT,
                                                        snap_id,
                                                        ordinal,
                                                        drop_off_rate
       
                                                    FROM snap_studio_story_snap_metric
                                                    WHERE  ordinal =0;
                                                    """
                                                    ) AS dr USING (story_id) 


                                                  --where date(interval_time)>current_date - 180 
                                                    order by name, interval_time asc
    
    
                  ) non_fin
                  )
                  SELECT *, 
                      -- CAST(story_id AS INT64) story_id_2
                  FROM cte
                  WHERE ranking <= 168
                  AND published_at >= '2022-01-01'
                  ORDER BY name ASC, story_id, ranking ASC;''')
  
    df = pd.read_gbq(sql_query, credentials = credentials)
    return df

@st.cache(ttl=43200)
def benchmark_data():
  sql_query2 = ('''WITH cte AS (SELECT
    non_fin.*,
    dense_rank() over(partition by non_fin.story_id order by interval_time) ranking

  FROM
  ( --- non-financial numbers that are not aggregated
    SELECT 
    name, 
    title,
    datetime(published_at,"America/Toronto") published_at,
    datetime(interval_time,"America/Toronto") interval_time,
    story_id,
    --metrics
    avg_time_spent_per_user avg_time_spent_per_user,
    completion_rate completion_rate,
    demo_age_18_to_17_diff demo_age_13_to_17_diff,
    demo_age_18_to_24_diff demo_age_18_to_24_diff,
    demo_age_25_to_34_diff demo_age_25_to_34_diff,
    demo_age_35_plus_diff demo_age_35_plus_diff,
    demo_age_unknown_diff demo_age_unknown_diff,
    demo_female_diff demo_female_diff,
    demo_male_diff demo_male_diff,
    demo_gender_unknown_diff demo_gender_unknown_diff,

    screenshots_diff screenshots_diff,
    shares_diff shares_diff,
    subscribers_diff subscribers_diff,
    tiles tiles,
    topsnap_views_diff topsnap_views_diff,
    topsnap_views topsnap_views_total,
    total_time_viewed_diff total_time_viewed_diff,
    total_views_diff total_views_diff,
    unique_completers_diff unique_completers_diff,
    unique_topsnap_views_diff unique_topsnap_views_diff,
    ---unique_topsnaps_per_user_diff unique_topsnaps_per_user_diff,
    sum(unique_topsnaps_per_user_diff) over (partition by story_id order by interval_time ) cum_unique_topsnaps_per_user_diff,
    unique_viewers_diff unique_viewers_diff,
    unique_viewers unique_viewers_total,
    drop_off_rate

  FROM `distribution-engine.post_time_series.snap_post_metrics_30_minutes_with_diff` hourly
  LEFT JOIN  EXTERNAL_QUERY(
     "projects/distribution-engine/locations/us/connections/postgres",
     """
    SELECT story_id::TEXT,
            published_at,
            title,
            name
    FROM snap_studio_story_metric
        LEFT JOIN snap_publishers USING (publisher_id)
    ---where name in ('Crafty')
        order by published_at desc
                """
        ) AS pub USING (story_id) 

         LEFT JOIN  EXTERNAL_QUERY(
     "projects/distribution-engine/locations/us/connections/postgres",
     """
    select
    story_id::TEXT,
       snap_id,
       ordinal,
       drop_off_rate
  from snap_studio_story_snap_metric
  where  ordinal =0;
                """
        ) AS dr USING (story_id) 


       --where date(interval_time)>current_date - 360 
    order by name, interval_time asc
    
    
    ) non_fin
    ), 
  cte_2 AS(
        SELECT name, 
        title, 
        published_at, 
        interval_time,
        story_id, 
        ranking,
  		topsnap_views_diff,
        topsnap_views_total,
  		unique_viewers_diff,
        unique_viewers_total
        FROM cte
        WHERE ranking in (24, 48, 72, 96, 120, 144, 168)
        )
  SELECT *,
    topsnap_views_total - LAG(topsnap_views_total) OVER (PARTITION BY name, story_id ORDER BY ranking) topsnap_daily_diff,
    unique_viewers_total - LAG(unique_viewers_total) OVER (PARTITION BY name, story_id ORDER BY ranking) unique_viewers_daily_diff
  FROM cte_2
  WHERE published_at >= current_date - 120;''')

  benchmarks = pd.read_gbq(sql_query2, credentials = credentials)
  return benchmarks

# Create Sidebar 
menu = ["Topsnap Forecast", "ML Test & Validate"]
choice = st.sidebar.selectbox("Menu", menu)

st.write("*Forecasting is powered by hourly BigQuery data, refreshed every 30 minutes*")

if choice == 'Topsnap Forecast':
    
    # Create dropdown-menu / interactive forecast graph
    st.write("# Forecasting Topsnaps")

    about_bar = st.expander("**About This Section**")
    about_bar.markdown("""
                        * The interactive chart below showcases the predicted forecast of topsnaps for your chosen episode (blue line) vs. actual values (white circle) as well as forecasted values into the future using a Neural Network.
                        * Input the episode's Story ID and number of hours you would like to forecast in the future.
                        * Click the "Forecast Topsnaps" button to run the model and visualize the results.

                        **NOTE: The number of hours to forecast should always remain at 24 hours or below - a general rule of thumb is that the number of hours forecasted should always be lower than the number of hours we currently have data for**
                       """)

    #Choose an episode 
    episode = st.text_input("Enter the Story ID here:", "")

    hour_choices = {24: '24', 48: '48', 72: '72', 96: '96', 120:'120', 144:'144', 168:'168'}
    #def format_func(hours):
      #return hour_choices[hours]

    hours = st.selectbox("Select the hourly window you would like to forecast to", options=list(hour_choices.keys()), format_func = lambda x:hour_choices[x])
    #hours = st.number_input("Enter the number of hours to forecast (24 hours or below)", 0, 24)
    
    forecast_total = st.button("Forecast Topsnaps - Total View")
    if forecast_total:
      df = update_data()
      benchmarks = benchmark_data()
      st.plotly_chart(forecast_totalview(episode, hours), use_container_width=True)

    forecast_daily = st.button("Forecast Topsnaps - Daily View")
    if forecast_daily:
      df = update_data()
      benchmarks = benchmark_data()
      st.plotly_chart(forecast_dailyview(episode, hours), use_container_width=True)

if choice == 'ML Test & Validate':

    # Create dropdown-menu / interactive forecast graph
    st.write("# Train and Test the Model")

    about_bar = st.expander("**About This Section:**")
    about_bar.markdown("""
                        **FOR BI PURPOSES ONLY**
                        * Use the features below to perform the following:
                            * Plot the training loss of the neural network model
                            * Display test performance (loss, MAE and RMSE)
                            * Perform 3 and 5 fold cross validation and display metrics (MASE, MAE, RMSE)
                       """)

    train_episode = st.text_input("Enter the Story ID here:", "")

    plot = st.button("Plot Loss")
    if plot:
      df = update_data()
      model = tts_model()
      st.line_chart(plot_loss(train_episode))

    testing_metrics = st.button ("Display Test Metrics")
    if testing_metrics:
      df = update_data()
      model = tts_model()
      st.dataframe(test_metrics(train_episode))

    cross_three = st.button("Cross-validate 3-folds")
    if cross_three:
      df = update_data()
      st.dataframe(crossvalidate_three(train_episode))

    cross_five = st.button("Cross-validate 5-folds")
    if cross_five:
      df = update_data()
      st.dataframe(crossvalidate_five(train_episode))
