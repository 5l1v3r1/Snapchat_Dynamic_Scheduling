import streamlit as st
import math
import pandas as pd
import numpy as np
from numpy.ma.core import log
from datetime import datetime, timedelta

from neuralprophet import NeuralProphet
import chart_studio.plotly as py
from plotly import graph_objs as go
from neuralprophet.benchmark import Dataset, NeuralProphetModel, SimpleExperiment, CrossValidationExperiment

from google.oauth2 import service_account
from google.cloud import bigquery

from st_aggrid import GridOptionsBuilder, AgGrid, GridUpdateMode, DataReturnMode, JsCode

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

#Ignore warning
st.set_option('deprecation.showPyplotGlobalUse', False)

#Creating Functions 

def forecast_totalview(choose_episode, choose_hours):
  #Load in episode
  this_episode_df = df[df['story_id'].isin([choose_episode])]
  this_episode_metrics = this_episode_df.loc[:, ['interval_time', 'topsnap_views']]

  data = this_episode_metrics.rename(columns = {'interval_time': 'ds', 'topsnap_views':'y'}).drop_duplicates(subset='ds').astype({'y' : 'int32'})

  #Get actual hours from time window, and actual hours from last value
  end_time = data['ds'].head(1)+timedelta(hours=choose_hours)
  last_time = data.tail(1)['ds'].values[0]

  # Get actual hours length 
  time_length = last_time - data['ds'].head(1)
  time_length = round(time_length / timedelta(hours=1)).astype('int')
  time_length = time_length.values[0]

  #Get steps to the actual chosen time window - determine forecasting length
  answer = end_time-last_time
  hours_number = round(answer / timedelta(hours=1)).astype('int')
  hours_number = hours_number.values[0]
  if time_length > choose_hours:
    hours_number = 0
  
  # Get steps to actual time window for retrospective views
  isolated_endtime = end_time.values[0]
  retro_window = data.ds.searchsorted(isolated_endtime)

  # Train and load model
  m = tts_model()
  
  metrics = m.fit(data, freq='H')
  
  future = m.make_future_dataframe(data, periods=hours_number, n_historic_predictions=len(data)) 
  prediction = m.predict(future)

  if time_length > choose_hours:
    prediction = prediction[:retro_window]

  #Get Confidence Intervals
  y = prediction['yhat1']
  average_data = []
  for ind in range(len(y)):
    average_data.append(np.mean(y[0:ind+1]))
  prediction['running_mean'] = average_data

  std_data = []
  for ind in range(len(y)):
    std_data.append(np.std(y[0:ind+1]))
  prediction['running_std'] = std_data

  prediction['n'] = prediction.index.to_list()
  prediction['n'] = prediction['n'] + 1
  prediction['ci'] = 1.96 * prediction['running_std'] / np.sqrt(prediction['n'])
  prediction['yhat_lower'] = prediction['yhat1'] - prediction['ci']
  prediction['yhat_upper'] = prediction['yhat1'] + prediction['ci']

  #Visualize Model 
  yhat = go.Scatter(x = prediction['ds'], 
                    y = prediction['yhat1'],
                    #y = prediction['yhat24'], 
                    mode = 'lines',
                    marker = {'color': 'blue'},
                    line = {'width': 4},
                    name = 'Forecast',
                    )
  yhat_lower = go.Scatter(x = prediction['ds'],
                          y = prediction['yhat_lower'],
                          marker = {'color': 'powderblue'},
                          showlegend = False,
                          #hoverinfo = 'none',
                          )
  yhat_upper = go.Scatter(x = prediction['ds'],
                          y = prediction['yhat_upper'],
                          fill='tonexty',
                          fillcolor = 'powderblue',
                          name = 'Confidence (95%)',
                          #hoverinfo = 'yhat_upper',
                          mode = 'none'
                          )
  
  actual = go.Scatter(x = prediction['ds'],
                      y = prediction['y'],
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
  
  layout_data = [yhat_lower, yhat_upper, yhat, actual]

  #Get Episode name
  episode_df = df[df['story_id'].isin([choose_episode])]
  episode_name = episode_df.head(1)['title'].values[0]

  #Get Channel name 
  channel_df = benchmarks[benchmarks['name'].isin(episode_df.name)]
  channel_name = channel_df.head(1)['name'].values[0]

  #Get Test CTR
  ctr = episode_df.head(1)['best_test_ctr'].values[0]
  if ctr is not None:
    ctr = f'{round(ctr*100, 2)}%'

  #Get hours for title 
  start2 = future.dropna().tail(1)['y'].values[0]
  end2 = prediction.tail(1)['yhat1'].values[0]
  number = round(end2-start2)

  start_end = prediction.tail(24)
  start = start_end.head(1)['yhat1'].values[0]
  end = start_end.tail(1)['yhat1'].values[0]
  end = round(end)
  last_24 = round(end-start)

  #Get recent 168hr benchmark
  banger_bench = channel_df.loc[channel_df['true_hour'] == 168, ['topsnap_views_total']]
  banger_bench = banger_bench['topsnap_views_total'].mean()*2

  #Get benchmarks
  def get_benchmarks(choose):
    b_channel = benchmarks[benchmarks['name'].isin(episode_df.name)]
    b_channel = b_channel.loc[b_channel['true_hour'] == choose, ['topsnap_views_total']]
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

  elif ((choose_hours > 168) and (choose_hours <= 192)):
    channel_bench = get_benchmarks(192)
    day = 'Day 8'

  elif ((choose_hours > 192) and (choose_hours <= 216)):
    channel_bench = get_benchmarks(216)
    day = 'Day 9'

  elif ((choose_hours > 216) and (choose_hours <= 240)):
    channel_bench = get_benchmarks(240)
    day = 'Day 10'

  trending = ((end-channel_bench)/channel_bench)*100
  if trending > 0:
    trending = f'+{round(trending)}% above'
  else:
    trending = f'{round(trending)}% below'

  fig = go.Figure(data= layout_data, layout=layout)

  fig.update_layout(title={'text': (f'<b>{episode_name} - {channel_name}</b><br><br><sup>Total Topsnap Prediction = <b>{end:,}</b> ({trending} Avg)<br>{day} Topsnap Prediction = <b>{round(last_24):,}</b><br>Test CTR = <b>{ctr}</b></sup>'),
                           'y':0.91,
                           'x':0.075,
                           'font_size':22})
  fig.add_hline(y=channel_bench, line_dash="dot", line_color='purple',
                annotation_text=(f"Channel Avg at {choose_hours}hrs: <b>{round(channel_bench):,}</b>"), 
              annotation_position="bottom right",
              annotation_font_size=14,
              annotation_font_color="purple"
             )
  fig.add_hline(y=banger_bench, line_dash="dot", line_color='gold',
                annotation_text="168hr Banger Benchmark", 
              annotation_position="bottom right",
              annotation_font_size=14,
              annotation_font_color="black"
             )
  return fig

def forecast_dailyview(choose_episode, choose_hours):
  #Load in episode
  this_episode_df = df[df['story_id'].isin([choose_episode])]
  this_episode_metrics = this_episode_df.loc[:, ['interval_time', 'topsnap_views']]

  data = this_episode_metrics.rename(columns = {'interval_time': 'ds', 'topsnap_views':'y'}).drop_duplicates(subset='ds').astype({'y' : 'int32'})

  #Get actual hours from time window, and actual hours from last value
  end_time = data['ds'].head(1)+timedelta(hours=choose_hours+1)
  last_time = data.tail(1)['ds'].values[0]

  # Get actual hours length
  time_length = last_time - data['ds'].head(1)
  time_length = round(time_length / timedelta(hours=1)).astype('int')
  time_length = time_length.values[0]

  #Get steps to the actual chosen time window to determine forecasting length
  answer = end_time-last_time
  hours_number = round(answer / timedelta(hours=1)).astype('int')
  hours_number = hours_number.values[0]
  if time_length > choose_hours:
    hours_number = 0
  
  # Get actual starting value for the nearest 24 hour window (for indexing)
  start_24 = end_time - timedelta(hours=26)
  test_start = start_24.values[0]
  final_start24 = data.ds.searchsorted(test_start)

  #Get actual ending value from the historical dataframe (retrospective)
  test_end = end_time.values[0]
  final_end = data.ds.searchsorted(test_end)
  
  def forecasting():
    # Train and load model
    m = tts_model()
    metrics = m.fit(data, freq='H')
  
    future = m.make_future_dataframe(data, periods=hours_number, n_historic_predictions=len(data)) 
    prediction = m.predict(future)

    #Daily dataframe
    daily_end = prediction['ds'].head(1)+timedelta(hours=choose_hours+1)
    dend_isolated = daily_end.values[0]
    dend_final = prediction.ds.searchsorted(dend_isolated)

    show_prediction = prediction[final_start24:dend_final]
    show_prediction['y_daily'] = ((show_prediction.loc[:, ['y']]) - (show_prediction.loc[:, ['y']].shift(+1))).cumsum()
    show_prediction['yhat_daily'] = ((show_prediction.loc[:, ['yhat1']]) - (show_prediction.loc[:, ['yhat1']].shift(+1))).cumsum()
    
    #Get Confidence Intervals
    y = show_prediction['yhat_daily']
    average_data = []
    for ind in range(len(y)):
      average_data.append(np.mean(y[0:ind+1]))
    show_prediction['running_mean'] = average_data

    std_data = []
    for ind in range(len(y)):
      std_data.append(np.std(y[0:ind+1]))
    show_prediction['running_std'] = std_data

    show_prediction = show_prediction.reset_index().drop(columns=['index'])
    show_prediction['n'] = show_prediction.index.to_list()
    show_prediction['n'] = show_prediction['n'] + 1
    show_prediction['ci'] = 1.96 * show_prediction['running_std'] / np.sqrt(show_prediction['n'])
    show_prediction['yhat_lower'] = show_prediction['yhat_daily'] - show_prediction['ci']
    show_prediction['yhat_upper'] = show_prediction['yhat_daily'] + show_prediction['ci']
    #show_prediction = show_prediction.iloc[1: , :]

    return show_prediction
  
  #Construct layout for forecasting
  show_prediction = forecasting()
 
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
                          #hoverinfo = 'none',
                          )
  yhat_upper = go.Scatter(x = show_prediction['ds'],
                          y = show_prediction['yhat_upper'],
                          fill='tonexty',
                          fillcolor = 'powderblue',
                          name = 'Confidence (95%)',
                          #hoverinfo = 'yhat_upper',
                          mode = 'none'
                          )
  
  actual = go.Scatter(x = show_prediction['ds'],
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
  
  layout_data = [yhat_lower, yhat_upper, yhat, actual]

  #Topsnap values for display
  f_start = show_prediction.dropna().tail(1)['y_daily'].values[0]
  f_end = show_prediction.tail(1)['yhat_daily'].values[0]
  number = round(f_end - f_start)
  last_24 = round(show_prediction.tail(1)['yhat_daily'].values[0])

  display = "Topsnap Prediction" 
  
  if hours_number == 0:
    retro_data = data[final_start24:final_end]
    retro_data['y_daily'] = ((retro_data.loc[:, ['y']]) - (retro_data.loc[:, ['y']].shift(+1))).cumsum()
    #retro_data = retro_data.iloc[1: , :]

    #Construct different layout
    y = go.Scatter(x = retro_data['ds'], 
                    y = retro_data['y_daily'],
                    mode = 'lines',
                    marker = {'color': 'black'},
                    line = {'width': 4},
                    name = 'Historical',
                    )
  
    actual = go.Scatter(x = retro_data['ds'],
                      y = retro_data['y_daily'],
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
  
    layout_data = [y, actual]

    #Topsnap values for display
    number = 0
    last_24 = retro_data.tail(1)['y_daily'].values[0]
    display = "Topsnap Performance"
    
  #Get Episode name
  episode_df = df[df['story_id'].isin([choose_episode])]
  episode_name = episode_df.head(1)['title'].values[0]

  #Get Channel name 
  channel_df = benchmarks[benchmarks['name'].isin(episode_df.name)]
  channel_name = channel_df.head(1)['name'].values[0]

  #Get Test CTR
  ctr = episode_df.head(1)['best_test_ctr'].values[0]
  if ctr is not None:
    ctr = f'{round(ctr*100, 2)}%'
    
  #Get benchmarks
  def get_benchmarks(choose):
    b_channel = benchmarks[benchmarks['name'].isin(episode_df.name)]
    b_channel = b_channel.loc[b_channel['true_hour'] == choose, ['topsnap_daily_diff']]
    channel_bench = b_channel['topsnap_daily_diff'].mean()

    return channel_bench

  if choose_hours <= 24:
    b_channel = benchmarks[benchmarks['name'].isin(episode_df.name)]
    b_channel = b_channel.loc[b_channel['true_hour'] == 24, ['topsnap_views_total']]
    channel_bench = b_channel['topsnap_views_total'].mean()
    day = 'Day 1'

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

  elif ((choose_hours > 168) and (choose_hours <= 192)):
    channel_bench = get_benchmarks(192)
    day = 'Day 8'

  elif ((choose_hours > 192) and (choose_hours <= 216)):
    channel_bench = get_benchmarks(216)
    day = 'Day 9'

  elif ((choose_hours > 216) and (choose_hours <= 240)):
    channel_bench = get_benchmarks(240)
    day = 'Day 10'

  #Perentage% Change for display
  trending = ((last_24-channel_bench)/channel_bench)*100
  if trending > 0:
    trending = f'+{round(trending)}% above'
  else:
    trending = f'{round(trending)}% below'

  #Visualize layout
  fig = go.Figure(data=layout_data, layout=layout)

  fig.update_layout(title={'text': (f'<b>{day} : {episode_name} - {channel_name}</b><br><br><sup>{day} {display} = <b>{round(last_24):,}</b> ({trending} Avg)<br>{hours_number-1:,}hr Topsnap Prediction = <b>{number:,}</b><br>Test CTR = <b>{ctr}</b></sup>'),
                           'y':0.91,
                           'x':0.075,
                           'font_size':22})
  fig.add_hline(y=channel_bench, line_dash="dot", line_color = 'purple',
                annotation_text=(f"{day} Channel Avg: <b>{round(channel_bench):,}</b>"), 
              annotation_position="bottom right",
              annotation_font_size=14,
              annotation_font_color="purple"
             )
  return fig

def tts_model():
    #Train and Test the  model
    m = NeuralProphet(daily_seasonality=False,
                      num_hidden_layers=2,
                    d_hidden=4,
                    seasonality_mode='muplicative', 
                    learning_rate=5.0,
                    batch_size=50,
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
    plot_metrics = plot_metrics.rename(columns={'MSELoss':'Train', 'MSELoss_val': 'Test'})
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
    final_test = metrics_train.tail(1)

    return final_test

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
          "batch_size":50,
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
          "batch_size":50,
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

#Summary Table Function 

@st.experimental_memo(ttl=3600)
def summary_table():
  #Function for rounding to 24 window
  def round_to_multiple(number, multiple):
    return multiple * math.ceil(number / multiple)

  #Get episodes currently running from each channel
  latest = df.loc[df.groupby('name').published_at.idxmax()]
  latest_df = df[df['story_id'].isin(latest.story_id)]
  latest_df = latest_df[~latest_df['name'].isin(['Ray Reacts', 'That Was Epic', 'Hacksmith'])]

  #Store episode info and create channel dictionary for looping
  channels = latest_df.name.unique()
  channels_dict = {elem : pd.DataFrame() for elem in channels}
  model_channel = channels_dict

  #Empty lists to store values from loop
  id_list = []
  episode_list = []
  channel_list = [] 

  forecast_list = []
  hours_running = []
  forecast_period = []
  benchmark_list = []
  trend_num_list = []
  ctr_list = []
  actual_list = []
  actual_benchmark = []
  trend_actual = []

  #Loop to train each channel's episode for forecasting, and storing individual values in each list
  for key in channels_dict.keys():
    channels_dict[key] = latest_df[:][latest_df.name == key]

    model_channel = channels_dict[key].loc[:, ['interval_time', 'topsnap_views']]
    model_channel = model_channel.rename(columns = {'interval_time': 'ds', 'topsnap_views':'y'})
    model_channel = model_channel.drop_duplicates(subset='ds')
    model_channel = model_channel.astype({'y' : 'int32'})

    #Get most recent timestamp
    last_time = model_channel.tail(1)['ds'].values[0]

    #Number of hours running of the current episode
    time_length = last_time - model_channel['ds'].head(1)
    data_length = round(time_length / timedelta(hours=1)).astype('int')
    data_length = data_length.values[0]

    #Nearest 24-hour window
    ending_hours = round_to_multiple(data_length, 24)

    #Number of hours to forecast
    hours = round(ending_hours - data_length)

    #Create and fit model 
    try:
      model = tts_model()
      metrics = model.fit(model_channel, freq="H")
      future = model.make_future_dataframe(model_channel, periods=hours, n_historic_predictions=len(model_channel)) 
      prediction = model.predict(future)

    except ValueError:
      continue
    
    #Append after try/except block 
    #ID
    id_df = df[df['story_id'].isin(channels_dict[key].story_id)]
    id = id_df.head(1)['story_id'].values[0]
    id_list.append(id)
    #Episode 
    episode_df = df[df['title'].isin(channels_dict[key].title)]
    episode = episode_df.head(1)['title'].values[0]
    episode_list.append(episode)
    #Channels
    channel_df = df[df['name'].isin(channels_dict[key].name)]
    channel = channel_df.head(1)['name'].values[0]
    channel_list.append(channel)

    #Append previous variables
    hours_running.append(data_length)
    forecast_period.append(ending_hours)

    #Store the last actual value
    last_actual = future.dropna().tail(1)['y'].values[0]
    actual_list.append(last_actual)

    #Store the channel benchmark at the actual value hour 
    actual_bench_df = benchmarks[benchmarks['name'].isin(channels_dict[key].name)]
    actual_bench = actual_bench_df.loc[actual_bench_df['true_hour']==data_length]
    actual_bench = actual_bench['topsnap_views_total'].mean()
    actual_benchmark.append(actual_bench)

    #Store the final forecasting value
    forecasts = prediction.tail(1)['yhat1'].values[0]
    forecast_list.append(forecasts)

    #Store the nearest 24 hour benchmark
    #bench = benchmarks[benchmarks['name'].isin(channels_dict[key].name)]
    bench = actual_bench_df.loc[actual_bench_df['true_hour'] == ending_hours]
    channel_bench = bench['topsnap_views_total'].mean()
    benchmark_list.append(channel_bench)

    #Store CTR
    ctr_df = latest[latest['name'].isin(channels_dict[key].name)]
    ctr = ctr_df['best_test_ctr'].values[0]
    #if ctr is not None:
      #ctr = round((ctr*100),2)
    ctr_list.append(ctr)

    #Store % Difference from Actual 
    trending_actual = ((last_actual - actual_bench) / actual_bench)
    trend_actual.append(trending_actual)

    #Store % Difference from Forecast 
    trending = ((forecasts - channel_bench) / channel_bench)
    trend_num_list.append(trending)
  
  #Create Summary df
  final_df = pd.DataFrame({'Story ID': id_list,
                         'Channel': channel_list,
                         'Episode': episode_list,
                         'Test CTR(%)': ctr_list,
                         'Topsnap Performance': actual_list,
                         'Hours Running': hours_running,
                         "Actual Hours Benchmark": actual_benchmark,
                         "Actual % Against Avg": trend_actual,
                         'Topsnap Forecast': forecast_list,
                         'Forecast Period': forecast_period,
                         'Channel Benchmark': benchmark_list,
                         'Forecast % Against Average': trend_num_list
                         })
  
  #Fix dtypes for formatting
  final_df['Test CTR(%)'] = final_df['Test CTR(%)'].replace('None', 0).astype('float')
  final_df = final_df.fillna(0)
  final_df['Test CTR(%)'] = final_df['Test CTR(%)'].replace(0, np.nan)
  final_df['Forecast % Against Average'] = final_df['Forecast % Against Average'].replace(0, np.nan)
  final_df['Channel Benchmark'] = final_df['Channel Benchmark'].replace(0, np.nan)

  #Create Decision logic
  final_df['Consideration'] = np.select(
    [   #Let It Ride
        #No shaba; between 48-96 hours and trending 25% or above
        (~final_df['Channel'].isin(['What The Fork!?', 'Snacks & Hacks', 'The Shaba Kitchen', 'The Pun Guys']))
        &(final_df['Forecast Period']>24) &(final_df['Forecast Period']<=96) 
        &(final_df['Forecast % Against Average']>=0.25)
        #Shaba; between 72 and 96 trending at our above 25%; 
        |(final_df['Channel'].isin(['What The Fork!?', 'Snacks & Hacks', 'The Shaba Kitchen', 'The Pun Guys'])) 
        &(final_df['Forecast Period']>=72) & (final_df['Forecast Period']<=96)
        &(final_df['Forecast % Against Average']>=0.25)
        #Any; between 120 and 168 trending +90% 
        |(final_df['Forecast Period']>=120) & (final_df['Forecast Period']<=168)
        &(final_df['Forecast % Against Average']>=0.9) 
        #Any between 168 and 196 hours and trending 150% or greater
        |(final_df['Forecast Period']>168) & (final_df['Forecast Period']<=192)
        &(final_df['Forecast % Against Average']>=1.5)
        #Any at 192 trending 150% or greater
        |(final_df['Forecast Period']==192)
        &(final_df['Forecast % Against Average']>=1.5)
        #Any between 216 and 240
        |(final_df['Forecast Period']>=216) & (final_df['Forecast Period']<=240)
        &(final_df['Forecast % Against Average']>=2.0),

      # Investigate - Bullish
      #Any; at 120 to 168 trending between 50% and 90%
      (final_df['Forecast Period']>=120) &(final_df['Forecast Period']>=168)
      &(final_df['Forecast % Against Average']>=0.5) & (final_df['Forecast % Against Average']<=0.9)
      #High CTR between 72 and 96 hours where % is positive and % is increasing (by 0.1)
      |(final_df['Test CTR(%)'] >=0.28)
      &(final_df['Forecast Period']>=72) & (final_df['Forecast Period']<=96)
      &(final_df['Forecast % Against Average']>0)
      &((final_df['Forecast % Against Average'] - final_df['Actual % Against Avg']) >= 0.1)
      #High CTR between 72 and 96 hours where % is negative, but % is increasing (by 0.2)
      |(final_df['Test CTR(%)'] >=0.28)
      &(final_df['Forecast Period']>=72) & (final_df['Forecast Period']<=96)
      &(final_df['Forecast % Against Average']<0) & (final_df['Actual % Against Avg']<0)
      &((final_df['Forecast % Against Average'] + final_df['Actual % Against Avg']) >= 0.2),
     
     #Investigate - Bearish 
     #Not Shaba; 48-96 hours trending -25% to -50%
     (~final_df['Channel'].isin(['What The Fork!?', 'Snacks & Hacks', 'The Shaba Kitchen', 'The Pun Guys']))
     &(final_df['Forecast Period']>=48) & (final_df['Forecast Period']<=96)
     &(final_df['Forecast % Against Average']<= -0.25) & (final_df['Forecast % Against Average']> -0.5)
     #Not Shaba; 48hours with high CTR and trending -50% or below
     |(~final_df['Channel'].isin(['What The Fork!?', 'Snacks & Hacks', 'The Shaba Kitchen', 'The Pun Guys']))
     &(final_df['Test CTR(%)'] >=0.28)
     &(final_df['Forecast Period']==48)
     &(final_df['Forecast % Against Average']<= -0.5)
     #Shaba; between 72 and 96 hours trending -25% to -50%
     |(final_df['Channel'].isin(['What The Fork!?', 'Snacks & Hacks', 'The Shaba Kitchen', 'The Pun Guys']))
     &(final_df['Forecast Period']>=72) & (final_df['Forecast Period']<=96)
     &(final_df['Forecast % Against Average']<= -0.25) & (final_df['Forecast % Against Average']> -0.5)
     #All shows; High CTR with -50% or lower % at 72
     |(final_df['Test CTR(%)'] >=0.28)
     &(final_df['Forecast Period']==72)
     &(final_df['Forecast % Against Average'] <= -0.50)
     # Any; between 72 and 96 where trend between -25% and +25% but is decreasing significantly
     |(final_df['Forecast Period']>=72) & (final_df['Forecast Period']<=96)
     &(final_df['Actual % Against Avg']> -0.25) &(final_df['Actual % Against Avg']< 0.25)
     & ((final_df['Forecast % Against Average'] - final_df['Actual % Against Avg']) <= -0.2)
     #Any between 72 to 96 hours, where trend is positive but % is decreasing (by 0.2)
     |(final_df['Forecast Period']>=72) & (final_df['Forecast Period']<=96)
     &(final_df['Actual % Against Avg']>0)
     &((final_df['Forecast % Against Average'] - final_df['Actual % Against Avg']) <= -0.2), 

     #Investigate - Average 
     #Any; between 72 and 96 hours trending anywhere from -25% to +25% (non inclusive)
     (final_df['Forecast Period']>=72) & (final_df['Forecast Period']<=96)
     &(final_df['Forecast % Against Average'] > -0.25)
     &(final_df['Forecast % Against Average'] < 0.25)
     #Any; 120 to 168  trending 0-50% above average
     |(final_df['Forecast Period']>=120)&(final_df['Forecast Period']<=168)
     &(final_df['Forecast % Against Average']> 0) & (final_df['Forecast % Against Average']< 0.5)
     #Non-Shaba; at 48 hours trending anywhere from -25% to +25%
     |(~final_df['Channel'].isin(['What The Fork!?', 'Snacks & Hacks', 'The Shaba Kitchen', 'The Pun Guys']))
     &(final_df['Forecast Period'] == 48)
     &(final_df['Forecast % Against Average'] > -0.25)
     &(final_df['Forecast % Against Average'] < 0.25)
     #Any at 192 between 100% and 150%
     |(final_df['Forecast Period'] == 192)
     &(final_df['Forecast % Against Average'] >= 1.0) &(final_df['Forecast % Against Average'] < 1.5)
     #Any between 216 and 240 trending 200% or greater 
     |(final_df['Forecast Period'] >= 216) &(final_df['Forecast Period'] <= 240)
     &(final_df['Forecast % Against Average'] < 2.0) &(final_df['Forecast % Against Average'] >= 1.5),
     
     #Replace It
     #Any; 120 to 168 trending 0% or less
     (final_df['Forecast Period']>=120) &(final_df['Forecast Period']<=168)
     &(final_df['Forecast % Against Average']<= 0)
     #No CTR at 72 trending below -50% 
     |(final_df['Test CTR(%)'].isna())
     &(final_df['Forecast Period']==72)
     &(final_df['Forecast % Against Average']<= -0.5)
     #Low CTR; at 72 trending -50% or below
     |(final_df['Test CTR(%)'] < 0.28)
     &(final_df['Forecast Period']==72)
     &(final_df['Forecast % Against Average']<= -0.5)
     #Any; at 96 and trending below -50%
     |(final_df['Forecast Period']==96)
     &(final_df['Forecast % Against Average']<= -0.5)
     #Non Shaba low ctr at 48 hours, trending below -50%
     |(~final_df['Channel'].isin(['What The Fork!?', 'Snacks & Hacks', 'The Shaba Kitchen', 'The Pun Guys']))
     &(final_df['Test CTR(%)'] < 0.28)
     &(final_df['Forecast Period'] ==48)
     &(final_df['Forecast % Against Average']<= -0.5)
     #Non Shab "" CTR is none
     |(~final_df['Channel'].isin(['What The Fork!?', 'Snacks & Hacks', 'The Shaba Kitchen', 'The Pun Guys']))
     &(final_df['Test CTR(%)'].isna())
     &(final_df['Forecast Period'] ==48)
     &(final_df['Forecast % Against Average']<= -0.5)
     #Any at 192 and below 100%
     |(final_df['Forecast Period'] == 192)
     &(final_df['Forecast % Against Average'] < 1.0)
     #Any between 216 and 240 below 150%
     |(final_df['Forecast Period'] >= 216) &(final_df['Forecast Period'] <= 240)
     &(final_df['Forecast % Against Average'] < 1.5)

    ],  

    ['Let It Ride', 
     'Investigate - Bullish', 
     'Investigate - Bearish',
     'Investigate - Average',
     'Replace It'
    ], 
    default='No Decision'
  )
  #Reorder df columns
  df_order = ['Story ID',
            'Channel',
            'Episode',
            'Consideration', 
            'Test CTR(%)',
            'Topsnap Performance',
            'Hours Running',
            'Actual Hours Benchmark',
            'Actual % Against Avg',
             'Topsnap Forecast',
             'Forecast Period',
             'Channel Benchmark',
             'Forecast % Against Average']

  #Create summary df, sort and reset index
  summary_df = final_df[df_order].sort_values(['Forecast % Against Average'], ascending=False)
  summary_df = summary_df.reset_index().drop(columns=['index'])

  return summary_df

#Create functions for conditional formatting
#Function for highlighting rows
def highlight_rows(row):
  value = row.loc['Consideration']
  if value == 'Let It Ride':
    color = '#BAFFC9' #Green
  elif value == 'Investigate - Bullish':
    color = '#BAE1FF' #Blue
  elif value == 'Investigate - Bearish':
    color = '#F4A460' # sandy brown
  elif value == 'Investigate - Average':
    color = '#FFFACD' #Lemon
  elif value == 'Replace It':
    color = '#FF6347'#tomato
  elif value == 'No Decision':
    color = '#F5F5F5' #white smoke
  return ['background-color: {}'.format(color) for r in row]

#Function for highlighting cells
def highlight_cells(val):
  if val >=1.0:
    color = '#00e673' #medium dark green
  elif val >= 0.5:
    color = '#66ffb3' #medium green
  elif val > 0:
    color = '#BAFFC9' #green
  elif val >= -0.25:
    color = '#ffc2b3' #lightred 
  elif val >= -0.80:
    color = '#ff8566' #tomato
  elif val < -0.80:
    color = '#ff471a' #red
  else:
    color = '#F5F5F5' #whitesmoke
  return 'background-color: {}'.format(color)

# Uses st.experimental_memo to only rerun when the query changes or after 30 min.
@st.experimental_memo(ttl=1800)
#@st.cache(ttl=1800)
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
    from snap_studio_story_snap_metric
    where  ordinal =0;
                """
        ) AS dr USING (story_id) 


       --where date(interval_time)>current_date - 180 
    order by name, interval_time asc
    
    
    ) non_fin
    )
    SELECT cte.*,
       split.best_test_ctr
    FROM cte
    LEFT JOIN EXTERNAL_QUERY(
                            "projects/distribution-engine/locations/us/connections/postgres",
                            """
                            WITH cte AS
                                      (
                                        SELECT *,
                                        ROUND(swipes::NUMERIC/paid_impressions, 3)      swipe_up_rate,
                                        ROUND(story_opens::NUMERIC/paid_impressions, 3) story_open_rate,
                                        ROUND(spend/1000000, 2)                true_spend,
                                        REGEXP_REPLACE(name, '[A-Z]$', '')        episode_name
                                        --(name, '[A-Z]$|A[A-Z]$', '')        episode_name
                                        FROM snap_marketing_ad_lifetime
                                        ORDER BY story_open_rate DESC
                                      )
                            SELECT episode_name title,
                            MAX(story_open_rate) best_test_ctr, 
                            MAX(swipe_up_rate) best_test_sur
                            FROM cte
                            GROUP BY episode_name
                            """
                        ) AS split       
        -- CAST(story_id AS INT64) story_id_2
    ON cte.title = split.title
    WHERE ranking <= 240
    AND published_at >= current_date - 90;''')
  
    df = pd.read_gbq(sql_query, credentials = credentials)
    return df

@st.experimental_memo(ttl=43200)
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
    where  ordinal =0
       """
        ) AS dr USING (story_id) 
       --where date(interval_time)>current_date - 360 
    order by name, interval_time asc
    ) non_fin
    ),
    cte_2 AS
    (SELECT cte.*,
        DATETIME_DIFF(interval_time, first_hour, HOUR) true_hour
    FROM cte
    LEFT JOIN (SELECT       story_id, 
                        ranking, 
                        interval_time first_hour
                FROM cte
                WHERE ranking = 1) start
    ON cte.story_id = start.story_id)
    SELECT cte_2.name, 
       cte_2.title, 
       cte_2.published_at, 
       cte_2.interval_time,
       cte_2.story_id, 
       cte_2.true_hour,
  	   cte_2.topsnap_views_diff,
       cte_2.topsnap_views_total,
  	   cte_2.unique_viewers_diff,
       cte_2.unique_viewers_total,
       daily.topsnap_views_total - LAG(daily.topsnap_views_total) OVER (PARTITION BY daily.name, daily.story_id ORDER BY daily.true_hour) topsnap_daily_diff,
       daily.unique_viewers_total - LAG(daily.unique_viewers_total) OVER (PARTITION BY daily.name, daily.story_id ORDER BY daily.true_hour) unique_viewers_daily_diff,
       split.best_test_ctr 
    FROM cte_2
    LEFT JOIN (SELECT name, 
                  title, 
                  --published_at, 
                  interval_time,
                  story_id, 
                  true_hour,
  		          topsnap_views_diff,
                  topsnap_views_total,
  		          unique_viewers_diff,
                  unique_viewers_total
            FROM cte_2
            WHERE true_hour in (24, 48, 72, 96, 120, 144, 168, 192, 216, 240))daily
    ON (cte_2.story_id = daily.story_id) AND (cte_2.true_hour = daily.true_hour)
    LEFT JOIN EXTERNAL_QUERY(
                            "projects/distribution-engine/locations/us/connections/postgres",
                            """
                            WITH cte AS
                                      (
                                        SELECT *,
                                        ROUND(swipes::NUMERIC/paid_impressions, 3)      swipe_up_rate,
                                        ROUND(story_opens::NUMERIC/paid_impressions, 3) story_open_rate,
                                        ROUND(spend/1000000, 2)                true_spend,
                                        REGEXP_REPLACE(name, '[A-Z]$', '')        episode_name
                                        --(name, '[A-Z]$|A[A-Z]$', '')        episode_name
                                        FROM snap_marketing_ad_lifetime
                                        ORDER BY story_open_rate DESC
                                      )
                            SELECT episode_name title,
                            MAX(story_open_rate) best_test_ctr, 
                            MAX(swipe_up_rate) best_test_sur
                            FROM cte
                            GROUP BY episode_name
                            """
                        ) AS split
    ON cte_2.title = split.title
    WHERE cte_2.published_at >= current_date - 90
      AND cte_2.true_hour <= 240
    ORDER BY title ASC, true_hour ASC
    ;''')

  benchmarks = pd.read_gbq(sql_query2, credentials = credentials)
  return benchmarks

# Create Sidebar 
menu = ["Episode Summary", "Topsnap Forecast", "ML Test & Validate"]
choice = st.sidebar.selectbox("Menu", menu)

st.write("*Forecasting is powered by NeuralProphet, and hourly data is derived from BigQuery - refreshed(cached) every 30 minutes*")

if choice == 'Episode Summary':
    # Create dropdown-menu / interactive forecast graph
    st.write("# Episode Summary - Currently Running Episodes")

    about_bar = st.expander("**About This Section**")
    about_bar.markdown("""
                        * Click the 'View Summary Table' below to see metrics and forecasts on all currently running episodes for all Snapchat channels
                        * "Considerations" are provided based on the current metrics and dynamic scheduling logic, however, it is strongly recommended that you use this table in conjunction with the Topsnap Forecasting tab to make scheduling decisions.

                        **NOTE: If the Channel doesn't appear in the table, then there is simply not enough running hours for its current episode to make a prediction yet - making a dynamic decision at this point would be irrelevant as it would be too early to assess performance** 
                       """)

    df = update_data()
    benchmarks = benchmark_data()

    #summary = st.button("View Summary Table")
    #if summary:
      #summary_df = summary_table()
      #st.dataframe(summary_df.style.apply(highlight_rows, axis=1).applymap(highlight_cells, subset=['Forecast % Against Average']).format(formatter={"Test CTR(%)": "{:.2%}", "Actual % Against Avg": "{:.2%}", "Forecast % Against Average": "{:.2%}", "Topsnap Performance": "{:,.0f}", 
      #"Topsnap Forecast": "{:,.0f}", "Actual Hours Benchmark": "{:,.0f}", "Channel Benchmark": "{:,.0f}"}))

    ag_chart = st.button("View Summary Table")
    if ag_chart:
      ag_df = summary_table()
      ag_df['Forecast % Against Avg'] = ag_df['Forecast % Against Average']

      percentages = ['Test CTR(%)', 'Actual % Against Avg', 'Forecast % Against Avg']
      values = ['Topsnap Performance', 'Actual Hours Benchmark', 'Topsnap Forecast', 'Channel Benchmark']

      ag_df['Forecast % Against Average'] = ag_df['Forecast % Against Average'].map("{:.2}".format).astype('float')
      for column in percentages:
        ag_df[column] = ag_df[column].map("{:.2%}".format)
        ag_df[column] = ag_df[column].replace('nan%', np.nan)
      for column in values:
        ag_df[column] = ag_df[column].map("{:,.0f}".format)

      jscode = JsCode("""
            function(params) {
                if (params.data.Consideration === 'Let It Ride') {
                    return {
                        'color': 'black',
                        'backgroundColor': '#BAFFC9'
                    }
                }
                if (params.data.Consideration === 'Investigate - Bullish') {
                    return {
                        'color': 'black',
                        'backgroundColor': '#BAE1FF'
                    }
                }
                if (params.data.Consideration === 'Investigate - Bearish') {
                    return {
                        'color': 'white',
                        'backgroundColor': '#F4A460'
                    }
                }
                if (params.data.Consideration === 'Investigate - Average') {
                    return {
                        'color': 'black',
                        'backgroundColor': '#FFFACD'
                    }  
                }
                if (params.data.Consideration === 'Replace It') {
                    return {
                        'color': 'white',
                        'backgroundColor': '#FF6347'
                    }  
                }
                if (params.data.Consideration === 'No Decision') {
                    return {
                        'color': 'black',
                        'backgroundColor': '#F5F5F5'
                    }  
                }
            };
            """)
      
      jscells = JsCode("""
            function (params) {
        
            if (params.data['Forecast % Against Average'] >=1.0) {
                return {
                        'color': 'white',
                        'backgroundColor': '#00e673'
                    }  
            }
            if (params.data['Forecast % Against Average'] >=0.5) {
                return {
                        'color': 'black',
                        'backgroundColor': '#66ffb3'
                    }
            }
            if (params.data['Forecast % Against Average'] > 0) {
                return {
                        'color': 'black',
                        'backgroundColor': '#BAFFC9'
                    }
            }
            if (params.data['Forecast % Against Average'] >=-0.25) {
                return {
                        'color': 'black',
                        'backgroundColor': '#ffc2b3'
                    }
            }
            if (params.data['Forecast % Against Average'] >=-0.8) {
                return {
                        'color': 'black',
                        'backgroundColor': '#ff8566'
                    }
            }
            if (params.data['Forecast % Against Average'] < -0.8) {
                return {
                        'color': 'white',
                        'backgroundColor': '#ff471a'
                    }
            }
            };
                      """)

      gb = GridOptionsBuilder.from_dataframe(ag_df)
      gb.configure_pagination(paginationAutoPageSize=True) #Add pagination
      gb.configure_side_bar() #Add a sidebar

      gb.configure_column('Forecast % Against Average', hide=True)
      gb.configure_column('Forecast % Against Avg', cellStyle=jscells)

      gridOptions = gb.build()
      gridOptions['getRowStyle'] = jscode
      
      grid_response = AgGrid(ag_df, 
                            gridOptions=gridOptions, 
                            allow_unsafe_jscode=True,
                            fit_columns_on_grid_load=True,
                            #data_return_mode='FILTERED_AND_SORTED', 
                            update_mode='NO_UPDATE', 
                            width='100%')

     #grid_response

if choice == 'Topsnap Forecast':
    
    # Create dropdown-menu / interactive forecast graph
    st.write("# Forecasting Topsnaps")

    about_bar = st.expander("**About This Section**")
    about_bar.markdown("""
                        * The interactive chart below showcases the predicted forecast of topsnaps for your chosen episode (blue line) vs. actual values (white circle) as well as forecasted values into the future using a Neural Network.
                        * Input the episode's Story ID and hourly forecast window
                        * Click the "Forecast Topsnaps" button to run the model and visualize the results - 'Total View' will display an episode's lifetime performance up to the hourly window you select, while 'Daily View' will only display performance of the past 24 hours from the window you select.

                        **NOTE: The number of hours to forecast should always remain at 24 hours or below - a general rule of thumb is that the number of hours forecasted should always be lower than the number of hours we currently have data for**
                       """)

    df = update_data()
    benchmarks = benchmark_data()

    #Choose an episode 
    episode = st.text_input("Enter the Story ID here:", "")

    hour_choices = {24: '24', 48: '48', 72: '72', 96: '96', 120:'120', 144:'144', 168:'168', 192:'192', 216:'216', 240:'240'}
    hours = st.selectbox("Select the hourly window you would like to forecast to", options=list(hour_choices.keys()), format_func = lambda x:hour_choices[x])
    
    forecast_total = st.button("Forecast Topsnaps - Total View")
    if forecast_total:
      st.plotly_chart(forecast_totalview(episode, hours), use_container_width=True)

    forecast_daily = st.button("Forecast Topsnaps - Daily View")
    if forecast_daily:
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
