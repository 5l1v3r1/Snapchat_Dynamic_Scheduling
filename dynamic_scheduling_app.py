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

#Allow to collect cache for update data function
@st.cache(suppress_st_warning=True, allow_output_mutation=True)

# Uses st.experimental_memo to only rerun when the query changes or after 10 min.
#@st.experimental_memo(ttl=600)

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

#Allow to collect cache for forecast function
@st.cache(suppress_st_warning=True, allow_output_mutation=True)

def get_forecast(choose_episode, choose_hours):
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

    prediction['yhat_lower'] = prediction['yhat1']*0.95
    prediction['yhat_upper'] = prediction['yhat1']*1.05

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

    episode_name = df[df['story_id'].isin([choose_episode])]
    episode_name = episode_name.head(1)['title'].values[0]

    start = future.dropna().tail(1)['y'].values[0]
    end = prediction.tail(1)['yhat1'].values[0]
    number = round(end-start)
  
    fig = go.Figure(data= data, layout=layout,
                  layout_title_text=(f'{episode_name} - {choose_hours}hr Topsnap Prediction<br>Predicted Topsnaps = {number:,}'))

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




# Create Sidebar 
menu = ["Topsnap Forecast", "ML Test & Validate"]
choice = st.sidebar.selectbox("Menu", menu)

st.write("*Forecasting is powered by hourly BigQuery data - refresh or re-open the webpage to update data when needed*")

if choice == 'Topsnap Forecast':
    
    # Create dropdown-menu / interactive forecast graph
    st.write("# Forecasting Topsnaps")

    about_bar = st.expander("**About This Section**")
    about_bar.markdown("""
                        * The interactive chart below showcases the predicted forecast of topsnaps for your chosen episode (blue line) vs. actual values (white circle) as well as forecasted values into the future using a Neural Network.
                        * Input the episode's Story ID number and number of hours you would like to forecast in the future for the model to run.
                        * Once the model has loaded, press the "Forecast Topsnaps" button to visualize the results.

                        **NOTE: The number of hours to forecast should always remain at 24 hours or below - a general rule of thumb is that the number of hours forecasted should always be lower than the number of hours we currently have data for**
                       """)

    #Choose an episode 
    episode = st.text_input("Enter the Story ID here:", "")
    hours = st.number_input("Enter the number of hours to forecast (24 hours or below)", 0, 24)
    
    forecast = st.button("Forecast Topsnaps")
    if forecast:
      df = update_data()
      st.plotly_chart(get_forecast(episode, hours), use_container_width=True)

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
