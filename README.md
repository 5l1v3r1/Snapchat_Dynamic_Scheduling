# Snapchat Dynamic Scheduling 
## Overview 
*This is a replication of a real-world project, edited for the suitability of a github repo*

Using Auto-ARIMA timeseries modelling to forecast social media video viewership performance. PELT change point detection (ruptures library) is layered on the forecast to identify major changes in trends in reference to actual performance + the model's prediction (offline change detection updated as the timeseries model updates).



## Purpose 
*This is a replication of a real-world project, edited for the suitability of a github repo*

Through previous analysis, it has been discovered that episodes on a social media platform (such as Snapchat) have their performance thwarted with the following episode's release on channel - with this, scheduling content to reflect performance rather than a weekly set schedule becomes an area of interest, i.e dynamically scheduling episodes to give longer running time to high performers, and cutting off time on platform for low performers.  

Predicting the future performance of an episode at incremental periods is valuable in that it can inform these dynamic scheduling decisions, especially when compared against benchmarks on channel. Furthermore, identifying major changes in trend is useful when looking to make real-time, performance-based scheduling decisions. 

The purpose of this web-app (as an analytics tool) is to help augment and expedite this data-informed process that may otherwise require rigorous analysis daily.



## ARIMA
Auto Regressive Integrated Moving Average using auto regression (p - predicting current value based on previous time periods' values), differencing (d), and error (q) to make future predictions is a widley used statistical technique in timeseries forecasting. The final version of the dynamic scheduling tool leverages BigQuery ML's Auto-ARIMA functionality to make non-seasonal predictions of video viewership performance in hourly intervals. 

Typically ARIMA models are quite reliable and more effective in making short term forecasting predictions vs other popular techniques such as Exponential Smoothing. Deep Learning options, of course, also exist (first iterations of this model utilizing FB Neural Prophet's AR-Net) but are often over-complicated and perform worse than their statistical counterparts. 

See [ds_app_2.py](https://github.com/a-memme/Snapchat_Dynamic_Scheduling/blob/main/ds_app_2.py) or the Model Performance section for model details.



## Streamlit Web-App
The following web-app utilizes streamlit-cloud to deploy several ML models (Auto-ARIMA; PELT cpd) created from different sources (BQML; Python ruptures) to provide functional, advanced analytics in the form of an internal tool. 

### Data 
Data is queried from a larger table (API ingestion every half hour) to isolate for the most recent episode for every channel and prepare the set to be loaded into an ARIMA forecasting model. Inclusions are made to ensure that the table is robust to changes in tactic/strategy such as single or multiple episode deletion. See below:
```
CREATE OR REPLACE TABLE `insert_table_here` AS
SELECT CAST(NULL AS TIMESTAMP) filled_time,
     CAST(NULL AS INT64) topsnap_views,
     CAST(NULL AS STRING) story_id_fixed;

FOR variable IN
(SELECT COUNT(DISTINCT post_id) story_counts,
      post_id,
      title,
      publisher_name
FROM (WITH cte AS(-- Account for "deleting" strategy to identify the currently running episode (could be 2nd not 1st episode)
                  -- Get Max interval time for the 2 most recent episodes per channel
                  SELECT MAX(interval_time) interval_time, 
                        published_at,
                        title, 
                        post_id,
                        publisher_name,
                        rolling_recency_ranking
                  FROM (--Rank all of the most recent episodes per channel from 1 and on
                        SELECT rt.account_id,
                                dt.account_name AS publisher_name, 
                                rt.platform, 
                                rt.post_id, 
                                datetime(dt.published_at, "America/Toronto") published_at,
                                datetime(rt.interval_time, "America/Toronto") interval_time, 
                                dt.title, 
                                dt.description, 
                                dt.tile_url, 
                                dt.logo_url,
                                rt.views, 
                                --LAG(rt.views) OVER(PARTITION BY rt.post_id ORDER BY rt.interval_time ASC) lag_views, 
                                rt.views - COALESCE((LAG(rt.views) OVER(PARTITION BY rt.post_id ORDER BY rt.interval_time ASC)), 0) views_diff, 
                                DENSE_RANK() OVER (PARTITION BY rt.account_id ORDER BY dt.published_at DESC) rolling_recency_ranking
                          FROM realtime_table rt
                          LEFT JOIN table_details dt
                          ON (rt.account_id = dt.account_id) AND (rt.post_id = dt.post_id)
                          WHERE TRUE 
                          AND rt.platform IN ('Snapchat')
                          ORDER BY post_id NULLS LAST, interval_time ASC
                          )
                  WHERE rolling_recency_ranking <= 20
                  GROUP BY published_at, title, post_id, rolling_recency_ranking, publisher_name
                  ORDER BY publisher_name, interval_time DESC, rolling_recency_ranking ASC
                  )
      -- Apply the final ranking to decide between the top 2 most recent episodes in a channel
      SELECT *, 
            DENSE_RANK() OVER(PARTITION BY publisher_name ORDER BY interval_time DESC, published_at DESC, rolling_recency_ranking ASC) final_rank
      FROM cte
      )
WHERE final_rank  = 1
AND publisher_name IS NOT NULL
GROUP BY post_id, title, publisher_name)

DO
INSERT INTO `insert_table_here`
SELECT timestamp(DATE_TRUNC(interval_time, HOUR)) AS filled_time,
       views topsnap_views, 
       post_id AS story_id_fixed
FROM `Views.snap_posts_rt`
WHERE TRUE 
AND post_id = variable.post_id
AND EXTRACT(MINUTE FROM interval_time) = 0
ORDER BY interval_time ASC;

END FOR;
```

### Summary Table
- Summary table compiles information regarding real-time video performance, timeseries forecasting data, changepoint detection data, daily changes in momentum (24 hour deltas), daily channel performance averages (90-day rolling), and hourly benchmarks to provide a high(er) level view on which episodes to keep running vs which to replace. Decisions are generated via conditional logic, informed by a combination of the metrics & values mentioned above.
- agGrid compatibility provides the ability to filter and select/unselect columns for scalability (episode names and ID's are discluded in the instance below)
- Data is cached periodically to save on computing power, and updated as data in the GCP database is updated.

![image](https://github.com/a-memme/Snapchat_Dynamic_Scheduling/assets/79600550/afac7728-0115-4bb2-9499-f759bc48fa55)


##### Current Section 
*i.e Current Hour, Current Perforance, Current Benchmark and % v Bench*
- represents how many hours the episode has been running for, its current performance (at that hour) and channel benchmark at that hour

##### Forecast Section 
*i.e Fcst Period, Forecast, Fcst Bench, and Fcst % against bench*
- represents the cumulative predicted performance of the episode at the forecasted hour (nearest 24-hour window), and how that relates to the channel benchmark at the respective forecasted hour.
- ARIMA forecast model is compiled in BigQuery ML using the following code (not attached as a sole file):
```
CREATE OR REPLACE MODEL `insert_model_name_here`
OPTIONS(MODEL_TYPE='ARIMA_PLUS',
       time_series_timestamp_col='filled_time',
       time_series_data_col='topsnap_views',
       TIME_SERIES_ID_COL = 'story_id_fixed',
       AUTO_ARIMA = TRUE,
       DATA_FREQUENCY = 'HOURLY',
       CLEAN_SPIKES_AND_DIPS = TRUE,
       ADJUST_STEP_CHANGES = TRUE,
       TREND_SMOOTHING_WINDOW_SIZE = 6,
       MAX_TIME_SERIES_LENGTH = 18,
       SEASONALITIES = ['NO_SEASONALITY']) 
       AS
SELECT * FROM `insert_table_here`
WHERE filled_time IS NOT NULL
ORDER BY story_id_fixed, filled_time ASC;
```

##### Trend Sentiment 
- Results of the changepoint detection model.
- 🔥 represents an increase in trend (in a recent time-frame - say past 48hrs for example) while a 🥶 represents a decrease in trend (in a recent timeframe).
- The number of emojis depicts the intensity of said trend. See "Dynamic Forecasting" section for more details.


### Forecasting + Momentum 
- Cumulative performance of an episode can be plotted using the respective story ID and 24 hour window in which we wish to forecast to (from the drop-down selection).
- The cumulative line graph shows the relevant benchmarks as well as areas in which positive or negative change has been detected depicted by 🔥 or 🥶 respectively (offline detection of the nearest 24 hour prediction).
- Historical performance is represented by the dark purple line while forecasted performance is represented by royal blue (See Legend). 

![image](https://github.com/a-memme/Snapchat_Dynamic_Scheduling/assets/79600550/530745a1-04c6-45bf-8ef2-3f52fdb3cc6f)




### Model Performance
- Evaluation of the model can be easily visualized in the webapp via the "Evaluate Model" button in the Model Performance Section.
- p, d, and q values generated by the Auto-ARIMA model as well as AIC results are visualized in the table. 
- Model testing is done internally and not shown below.

![image](https://github.com/a-memme/Snapchat_Dynamic_Scheduling/assets/79600550/9493e53b-15ab-478d-b4fa-f194f0101d45)


