# Snapchat Dynamic Scheduling 
## Overview 
*This is a replication of a real-world project, edited for the suitability of a github repo*

Using Auto-ARIMA timeseries modelling to forecast social media video viewership performance. Layering PELT change point detection to identify major changes in trends in reference to the model's prediction (offline change detection updated as the timeseries model updates).



## Purpose 
*This is a replication of a real-world project, edited for the suitability of a github repo*

Through previous analysis, it has been discovered that episodes on a social media platform (such as Snapchat) have their performance thwarted with the following episode's release on channel - with this, scheduling content to reflect performance rather than a weekly set schedule becomes an area of interest, i.e dynamically scheduling episodes to give longer running time to high performers, and cutting off time on platform for low performers.  

Predicting the future performance of an episode at incremental periods is valuable in that it can inform these dynamic scheduling decisions, especially when compared against benchmarks on channel. Furthermore, identifying major changes in trend is useful when looking to make real-time, performance-based scheduling decisions. 

The purpose of this web-app (as an analytics tool) is to help expedite this data-informed process that may otherwise require rigorous analysis daily.



## ARIMA
Auto Regressive Integrated Moving Average using auto regression (p - predicting current value based on previous time periods' values), differencing (d), and error (q) to make future predictions is a widley used statistical technique in timeseries forecasting. The final version of the dynamic scheduling tool leverages BigQuery ML's Auto-ARIMA functionality to make non-seasonal predictions of video viewership performance in hourly intervals. 
Typically ARIMA models are quite reliable and more effective in making short term forecasting predictions vs other popular techniques such as Exponential Smoothing. Deep Learning options, of course, also exist (first iterations of this model utilizing FB Neural Prophet's AR-Net) but are often over-complicated and perform worse than their statistical counterparts. 

See [ds_app_2.py](https://github.com/a-memme/Snapchat_Dynamic_Scheduling/blob/main/ds_app_2.py) or the Model Performance section for model details.



## Streamlit Web-App

### Summary Table
Summary table compiles information regarding real-time video performance, timeseries forecasting data, changepoint detection data, daily changes in momentum (24 hour deltas), daily channel performance averages (90-day rolling), and hourly benchmarks to provide a high(er) level view on which episodes to keep running vs which to replace. Decisions are generated via conditional logic, informed by a combination of the metrics & values mentioned above.

The Summary Table can be accessed by pressing the "View Summary Table" button (see picture below). Data is cached periodically to save on computing power, and updated as data in the GCP database is updated.

![image](https://github.com/a-memme/Snapchat_Dynamic_Scheduling/assets/79600550/efe6eae7-233b-41fc-87b9-c397a64a45db)

##### Current Section 
*i.e Current Hour, Current Perforance, Current Benchmark and % v Bench*
- represents how many hours the episode has been running for, its current performance (at that hour) and channel benchmark at that hour

##### Forecast Section 
*i.e Fcst Period, Forecast, Fcst Bench, and Fcst % against bench 
- represents the cumulative predicted performance of the episode at the forecasted hour (nearest 24-hour window), and how that relates to the channel benchmark at the respective forecasted hour.

##### Trend Sentiment 
- the results of the changepoint detection model.
- 🔥 represents an increase in trend (in a recent time-frame - say past 48hrs for example) while a 🥶 represents a decrease in trend (in a recent timeframe). The number of emojis depcits the intensity of said trend. See "Dynamic Forecasting" section for more details.


### Forecasting + Momentum 
Forecasting and momentum section offers an opprtunity to dive deeper into the data by plotting the cumulative performance (Plotly) of an episode using the respective story ID and 24 hour window in which we wish to forecast to (from the drop-down selection). The cumulative line graph shows the relevant benchmarks as well as areas in which positive or negative change has been detected depicted by 🔥 or 🥶 respectively (offline detection of the nearest 24 hour prediction). 

![image](https://github.com/a-memme/Snapchat_Dynamic_Scheduling/assets/79600550/a80f8738-fc43-4699-8c52-a3029bb63483)



Click the "Momentum Table" button to generate a table showcasing performance by day + momentum - a simplfiied, overall depiction of an episode's daily performance over time since inception:

![image](https://github.com/a-memme/Snapchat_Dynamic_Scheduling/assets/79600550/a057c53c-d08d-4c72-9802-454665d6b1d0)




### Train & Testing Model 
A tab is created for the data team to continually monitor performance of the model with ease. Validating the model through plot loss visualization as well as rendering key performance metrics (Loss, MAE, RMSE) is available with use of the "Plot Loss" and "Test Metrics" buttons. Quick cross-validation is also available using 3 and 5 folds respectively. 
