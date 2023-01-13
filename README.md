# Snapchat Dynamic Scheduling 
*Exploring different timeseries models to predict Snapchat video viewership - namely FB NeuralProphet's AR-Net and BigQuery's AutoARIMA models*

## Purpose 
*This is a replication of a real-world project, edited for the suitability of a github repo*

Through previous analysis, it has been discovered that episodes on a social media platform (such as Snapchat) have their performance thwarted with the following episode's release on channel - with this, scheduling content to reflect performance rather than a weekly set schedule becomes an area of interest, i.e dynamically scheduling episodes to give longer running time to high performers, and cutting off time on platform for low performers.  

Predicting the future performance of an episode at incremental periods is valuable in that it can inform these dynamic scheduling decisions, especially when compared against benchmarks on channel. The purpose of this dashboard & tool is to help expedite this data-informed process that may otherwise require rigorous analysis daily.

## Neural Prophet
[NeuralProphet (2020)](https://github.com/ourownstory/neural_prophet/?utm_source=hootsuite&utm_medium&utm_term&utm_content&utm_campaign&fbclid=IwAR1G35yRHAhO-UwiuR2UPGKwBlUtU98cJyPxu5vA4P-XTDzgBEwLe5Iq0EA) is an open-source time-series foreasting library released by Meta's Data Science team as an extension of their 2017 release of Facebook Prophet, incoporating auto-regressive (AR) deep-learning to its easy to use & interpretable framework. 

The current use of this model doesn't leverage the AR-net properties due to the better generalizability of the solo neural net for this particular task - yielding better results in many cases (See the Neural_Network Folder), and more substantially, providing more flexibility in its use (adding an auto regressive component makes the model more rigid within the NeuralProphet framework when lags are needed to be specified & fixed forecasting horizons are set). 

The NeuralProphet library is also useful in this particular case for a number of other benefits such as its extensiveness in hyper-parameter tuning, powerful PyTorch backend, robustness to missing data (uncontrollable due to fluctuation in Snapchat reporting delays), and suitability toward true future forecasting (See Neural_Network Folder). Furthermore, its generally low-code framework is very straightforward in making future predictions, where this process can become very cumbersome in many other deep learning libraries (very often we see models only testing on unseen data without the addition of making predictions into the future as well).

## ARIMA, Auto-ARIMA and Statistical Models

## Streamlit Dashboard 
### Summary Table
### Dynamic Forecasting 

### Train & Testing Model 
A tab is created for the data team to continually monitor performance of the model with ease. Validating the model through plot loss visualization as well as rendering key performance metrics (Loss, MAE, RMSE) is available with use of the "Plot Loss" and "Test Metrics" buttons. Quick cross-validation is also available using 3 and 5 folds respectively. 
