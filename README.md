# Uber-Pickup-analysis-using-Spatial-Temporal-Analysis-and-Geo-Spatial-Clustering (Streamlit Web App)
Uber Pickup analysis using Spatial Temporal Analysis &amp; Geo-Spatial Clustering

* This project explores the spatio-temporal patterns of taxi-service apps and taxi pick-up data and uses geo spatial clustering to make decisions from calculating pricing to finding the optimal positioning of cabs/drivers to maximize profits of the cab-share business. 

![Dashboard Demo](Demo/demo1.gif)
# Business Need:
* According to Gartner, by 2022, more than a  quarter billion connected vehicles will form a major element of the Internet of Things. Connected vehicles are projected to generate 25GB of data per hour, which can be analyzed to provide real-time monitoring and apps, and will lead to new concepts of mobility and vehicle usage.

* With the emerging app-based on demand taxi services , the competition in the market is increasing. Thus companies are using quantitative analysis of their app and taxi demands for neighborhoods of cities.

# Major Analysis Points to be answered:
1. Spatio-temporal analysis :Exploring trip data
* Getting inference about the number of trips per hour/day/week/Month.
* Number of trips completed per cab
* How different base stations are performing each month.
* Which base stations are best for different perspectives like businesses - may help us do price surge because of demand.

2. Geo Spatial Clustering: A better perspective:
* Use clustering techniques to find various spatial hotspots.
* Strategically place the driver’s in good locations(within these clusters) where in probability of getting a ride request are huge.
* optimal placing of their vehicles at different time of the day.
* Use these centroids for optimal pricing by analyzing which cluster deals with maximum requests, peak times etc.

3. Business analysis with Competitors & Active Vehicles Analysis

4. Web based Dashboard (Built Using Streamlit)



# Data:
Used public uber trip dataset to discuss building a real-time example for analysis and monitoring of car GPS data. The Uber trip dataset, which contains data generated by Uber from New York City.
Source : <a href="https://www.kaggle.com/fivethirtyeight/uber-pickups-in-new-york-city/notebooks?sortBy=hotness&group=everyone&pageSize=20&datasetId=360">Kaggle</a>


