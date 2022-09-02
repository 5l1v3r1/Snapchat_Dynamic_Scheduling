# Snapchat Dynamic Scheduling 
*Predicting Snapchat episode performance through a Neural Network*

## Purpose 
*This is a replication of a real-world analysis, edited for the suitability of a github repo*

Through previous analysis, it has been discovered that episodes on a social media platform (such as Snapchat) have their performance thwarted with the following episode's release on channel - with this, scheduling content to reflect performance rather than a weekly set schedule becomes an area of interest, i.e dynamically scheduling episodes to give longer running time to high performers, and cutting off time on platform for low performers.  

Predicting the future performance of an episode at incremental periods is valuable in that it can inform these dynamic scheduling decisions, especially when compared against benchmarks on channel. The purpose of this dashboard & tool is to help expedite this data-informed process that may otherwise require rigorous analysis daily.

## Neural Prophet
[NeuralProphet (2020)](https://github.com/ourownstory/neural_prophet/?utm_source=hootsuite&utm_medium&utm_term&utm_content&utm_campaign&fbclid=IwAR1G35yRHAhO-UwiuR2UPGKwBlUtU98cJyPxu5vA4P-XTDzgBEwLe5Iq0EA) is an open-source time-series foreasting library released by Meta's Data Science team as an extension of their 2017 release of Facebook Prophet, incoporating auto-regressive (AR) deep-learning to its easy to use & interpretable framework. 

The current use of this model doesn't leverage the AR-net properties of the model per-se, however its flexibility in hyper-paramter tuning, powerful PyTorch backend, and suitability toward true future forecasting lent toward an appropriate model for the task, with good performance results (See Neural_Network Folder). NeuralProphet framework makes it very straightforward to calculate predictions for future dates, whereas the same is not so straightforward when considering other popular options for time-series predictive modelling such as LSTM.

## Streamlit Dashboard 
### Dynamic Forecasting 

### Train & Testing Model 
A tab is created for the data team to also use this tool themselves to quickly and easly monitor the performance of the model. Validating the model through plot loss visualizing, rendering key metrics is available with use of the "Plot Loss" and "Test Metrics" buttons. Quick cross-validation is also available using 3 and 5 folds respectively. 
