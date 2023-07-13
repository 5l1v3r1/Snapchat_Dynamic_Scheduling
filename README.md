# Snapchat Dynamic Scheduling 
# Overview 
*This is a replication of a real-world project, edited for the suitability of a github repo*

Using Auto-ARIMA timeseries model to forecast social media video viewership performance. Layering PELT change point detection to identify major changes in trends in reference to the model's prediction (offline change detection updated as the timeseries model updates).

## Purpose 
*This is a replication of a real-world project, edited for the suitability of a github repo*

Through previous analysis, it has been discovered that episodes on a social media platform (such as Snapchat) have their performance thwarted with the following episode's release on channel - with this, scheduling content to reflect performance rather than a weekly set schedule becomes an area of interest, i.e dynamically scheduling episodes to give longer running time to high performers, and cutting off time on platform for low performers.  

Predicting the future performance of an episode at incremental periods is valuable in that it can inform these dynamic scheduling decisions, especially when compared against benchmarks on channel. Furthermore, identifying major changes in trend is useful when looking to make real-time, performance-based scheduling decisions. 

The purpose of this web-app (as an analytics tool) is to help expedite this data-informed process that may otherwise require rigorous analysis daily.

## ARIMA
Auto Regressive Integrated Moving Average using auto regression (p - predicting current value based on previous time periods' values), differencing (d), and error (q) to make future predictions is a widley used statistical technique in timeseries forecasting. The final version of the dynamic scheduling tool leverages BigQuery ML's Auto-ARIMA functionality to make non-seasonal predictions of video viewership performance in hourly intervals. 
Typically ARIMA models are quite reliable and more effective in making short term forecasting predictions vs other popular techniques such as Exponential Smoothing. Deep Learning options, of course, also exist (first iterations of this model utilizing FB Neural Prophet's AR-Net) but are often over-complicated and perform worse than their statistical counterparts. 

## Streamlit Web-App
The web app (replication of internal analytics tool) has 3 sections:
1. the summary table offering a high(er) level view of video performance with recommedations on scheduling decisions includes (combining a series of rules based on collected analytics on forecasting, change point detection, daily "momentum", and hourly benchmarks via conditional logic)
### Summary Table
Summary table compiles information regarding real-time video performance, timeseries forecasting data, changepoint detection data, daily changes in momentum (24 hour deltas), and hourly benchmarks to provide a high(er) level view on which episodes to keep running vs which to replace. Decisions are generated via conditional logic, informed by a combination of the metrics & values mentioned above.

![image](https://github.com/a-memme/Snapchat_Dynamic_Scheduling/assets/79600550/efe6eae7-233b-41fc-87b9-c397a64a45db)


### Dynamic Forecasting 

### Train & Testing Model 
A tab is created for the data team to continually monitor performance of the model with ease. Validating the model through plot loss visualization as well as rendering key performance metrics (Loss, MAE, RMSE) is available with use of the "Plot Loss" and "Test Metrics" buttons. Quick cross-validation is also available using 3 and 5 folds respectively. 
