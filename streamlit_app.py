#########
import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import pydeck as pdk
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.image as mpimg
#The matplotlib basemap toolkit is a library for plotting 2D data on maps in Python
from mpl_toolkits.basemap import Basemap
from matplotlib import cm #Colormap

#Animation Modules
from matplotlib.animation import FuncAnimation
import matplotlib.animation as animation


# SETTING PAGE CONFIG TO WIDE MODE
st.set_page_config(layout="wide")



from streamlit_folium import folium_static
import folium




def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)




# LOADING DATA
DATE_TIME = "date/time"






@st.cache(allow_output_mutation=True)
def load_data(nrows):
	#Load the datasets
	df_apr14=pd.read_csv("dataset/uber-raw-data-apr14.csv",nrows=nrows)
	df_may14=pd.read_csv("dataset/uber-raw-data-may14.csv",nrows=nrows)
	df_jun14=pd.read_csv("dataset/uber-raw-data-jun14.csv",nrows=nrows)
	df_jul14=pd.read_csv("dataset/uber-raw-data-jul14.csv",nrows=nrows)
	df_aug14=pd.read_csv("dataset/uber-raw-data-aug14.csv",nrows=nrows)
	df_sep14=pd.read_csv("dataset/uber-raw-data-sep14.csv",nrows=nrows)
	vehicle_data = pd.read_csv("dataset/Uber-Jan-Feb-FOIL.csv")
	data = df_apr14.append([df_may14,df_jun14,df_jul14,df_aug14,df_sep14], ignore_index=True)
	data_spatial = df_apr14.append([df_may14,df_jun14,df_jul14,df_aug14,df_sep14], ignore_index=True)
	data1 = df_sep14.loc[:50000,:]
	lowercase = lambda x: str(x).lower()
	data.rename(lowercase, axis="columns", inplace=True)
	data1.rename(lowercase, axis="columns", inplace=True)
	data[DATE_TIME] = pd.to_datetime(data[DATE_TIME])
	data1[DATE_TIME] = pd.to_datetime(data1[DATE_TIME])



	return data,data1,data_spatial,vehicle_data

local_css("style.css")


df,data1,df_spatial,vehicle_data = load_data(1000000)




def count_cols(cols):
	    return len(cols)

###################################################
def temporal():
	st.title("TEMPORAL ANALAYSIS")

	# temporal:
	# df = data
	df['Month'] = df[DATE_TIME].dt.month_name()
	df['Weekday'] = df[DATE_TIME].dt.day_name()
	df['Day'] = df[DATE_TIME].dt.day
	df['Hour'] = df[DATE_TIME].dt.hour
	df['Minute'] = df[DATE_TIME].dt.minute

	##1
	#Grouping by Hour 
	df_hour_grouped = df.groupby(['Hour']).count()

	#Creating the sub dataframe
	df_hour = pd.DataFrame({'Number_of_trips':df_hour_grouped.values[:,0]}, index = df_hour_grouped.index) 
	df_hour['Hour'] = df_hour.index

	c = alt.Chart(df_hour).mark_bar().encode(
	    alt.X('Hour', title='hour'),
	    alt.Y('Number_of_trips',title='Number_of_trips')
	).properties(
		title="Trip by hour")



	### 2 

	#Grouping by Weekday
	df_weekday_grouped = df.groupby(['Weekday'], sort = False).count()

	#Creating the grouped DataFrame
	df_weekday = pd.DataFrame({'Number_of_trips':df_weekday_grouped.values[:,0]}, index = df_weekday_grouped.index) 
	df_weekday['Weekday'] = df_weekday.index

	d = alt.Chart(df_weekday).mark_bar().encode(
	    alt.X('Weekday', title='Weekday'),
	    alt.Y('Number_of_trips',title='Number_of_trips')
	).properties(
		title="Trip by Weekday")



	#### 3
	#Grouping by Hour and Month
	df_hour_month_grouped = df.groupby(['Hour','Month']).count()

	#Creating the grouped DataFrame
	df_hour_month = pd.DataFrame({'Number_of_trips':df_hour_month_grouped.values[:,1]}, index = df_hour_month_grouped.index) 


	df_hour_month.reset_index(inplace= True)

	data_hour_month = df_hour_month['Number_of_trips'].values.reshape(24,6)

	df_hour_month = pd.DataFrame(data = data_hour_month, index = df_hour_month['Hour'].unique(), columns = df['Month'].unique())


# st.write(df_hour_month_grouped.head(10))

	# LAYING OUT THE TOP TEMPORAL SECTION OF THE APP
	col11,col12,col13 = st.beta_columns((2,1,2))


	with col11:
		container = st.beta_container()
		container.subheader("Trip by Hour")
		container.altair_chart(c,use_container_width=True)
		with container.beta_expander("See Insight"):
			st.write(""" 
						     1. We observe that the number of trips are higher around 16:00 and 18:00, with a spike at 17:00.
						     2. It matches the end of a working day in the United States (16:30), the time when the workers go home.
						     3. We can say that a big part of Uber's clients are workers.
						
						""")

	with col12:
		container1 = st.beta_container()
		container1.subheader("Trip by Weekday")
		container1.altair_chart(d,use_container_width=True)
		with container1.beta_expander("See Insight"):
			st.write("The lowest number of trips by weekday is 29k trip, that corresponds to Sunday.")

	with col13:
		st.subheader("Trips by hour and month")
		st.bar_chart(df_hour_month)


	#################################################




	###1

	

	df_hour_weekday = df.groupby('Hour Weekday'.split(), sort = False).apply(count_cols).unstack()
	plt.figure(figsize = (12,8))
	fig1 , ax = plt.subplots()
	ax = sns.heatmap(df_hour_weekday, cmap=cm.YlGnBu, linewidth = .5)





	###2


	#Grouping by Hour and weekday
	df_weekday_hour_grouped = df.groupby(['Weekday','Hour'], sort = False).count()

	#Creating the grouped DataFrame
	df_weekday_hour = pd.DataFrame({'Number_of_trips':df_weekday_hour_grouped.values[:,1]}, index = df_weekday_hour_grouped.index) 



	#Reseting the Index
	df_weekday_hour.reset_index(inplace= True)

	#Preparing the Number of trips data
	data_weekday_hour = df_weekday_hour['Number_of_trips'].values.reshape(7,24)



	# create dataframe


	# f = alt.Chart(df_weekday_hour).mark_bar().encode(
	#    column=alt.Column('Weekday'),
	#    x=alt.X('Hour'),
	#    y=alt.Y('Number_of_trips'),
	#    color=alt.Color('Hour',scale=alt.Scale(scheme='dark2'))
	# ).properties(
	#     width=130,
	#     height=130
	# )


	# from matplotlib.ticker import ScalarFormatter, FormatStrFormatter
	# figb , axp = plt.subplots()
	# figb.set_size_inches(6, 6)
	# sns.set_style('whitegrid')
	# axp = sns.pointplot(x="Hour", y="Number_of_trips", hue="Weekday", data=df_weekday_hour)
	# handles,labels = axp.get_legend_handles_labels()
	# #reordering legend content
	# handles = [handles[1], handles[5], handles[6], handles[4], handles[0], handles[2], handles[3]]
	# labels = [labels[1], labels[5], labels[6], labels[4], labels[0], labels[2], labels[3]]
	# axp.yaxis.set_major_formatter(FormatStrFormatter('%.0f'))
	# axp.set_xlabel('Hour', fontsize = 6)
	# axp.set_ylabel('Count of Uber Pickups', fontsize = 6)
	# axp.set_title('Hourly Uber Pickups By Day of the Week in NYC', fontsize=8)
	# axp.tick_params(labelsize = 8)
	# axp.legend(handles,labels,loc=0, title="Legend", prop={'size':8})
	# axp.get_legend().get_title().set_fontsize('6')
	
	





	
	
	# LAYING OUT THE MIDDLE TEMPORAL SECTION OF THE APP




	st.write(" ")
	col21,col22 = st.beta_columns((3,2))

	with col21:
		img=mpimg.imread('pickup.png')
		st.image(img,width=720,height=520)
		with st.beta_expander("Key Points:"):
			st.write("""1. We see that in working days there's a pulse at 7:00 and 8:00, it corresponds to the hour where the employees go to work. **(Price Surge)**""") 
			st.write("""2. This pulse is not present on weekend days. At the same time we see that on weekend days the number of trips around midnight, 1:00 and 2:00 is higher than on working days.""")
			st.write("""3. We can see that on working days (From Monday to Friday) the number of trips is higher from 16:00 to 21:00.On Friday the number of trips remains high until 23:00 and continues on early Saturday. It corresponds to the time where people come out from work, then go out for dinner or drink before the weekend.""")
			st.write("""4. We can notice the same pattern on Saturday, people tend to go out at night, the number of trips remains on high until early Sunday **(Price Surge on Weekends)**
			""")
		

	with col22:
		img=mpimg.imread('hour_day.png')
		st.image(img,width=520,height=520)

		
	


		




# SPATIAL MAP:
def base():
	st.title("Spatial Data Visualisation")


	#BASE ANALYSIS:
	base = df_spatial['Base'].value_counts()

	base = base.to_frame()
	#Creating the sub dataframe
	base.rename(columns = {'Base':'Number_of_trips'}, inplace = True) 
	base['base'] = base.index

	b1 = alt.Chart(base).mark_bar().encode(
	    alt.X('base', title='base'),
	    alt.Y('Number_of_trips',title='Number_of_trips')
	).properties(
		title="Trip by Bases")




	df_base_weekday = df.groupby('Weekday base'.split(), sort = False).apply(count_cols).unstack()
	plt.figure(figsize = (12,8))
	figb , axb = plt.subplots()
	axb = sns.heatmap(df_base_weekday, cmap=cm.YlGnBu, linewidth = .5)


	#################################################
	col_b1,col_b2,col_b3 = st.beta_columns((2,2,2))

	with col_b1:
		st.altair_chart(b1,use_container_width=True)
		with st.beta_expander("Identified regions:"):
			st.write(
				""" 
					* B02617 base had the highest number of trips
					* B02512 and B02764 are not the best bases from business perspective.
					* September was the best month for B02617
					* May was the best month for B02598
					* Apr was the best month for B02682
					* Thursday was the best day for B02598, B02617 and B02682


				""")

	with col_b2:
		img=mpimg.imread('baseW.png')
		st.image(img, caption='HeatMap by Weekday and Bases',width=520)

	with col_b3:
		img=mpimg.imread('baseM.png')
		st.image(img, caption='HeatMap by Month and Bases',width=520)



#Setting up the limits
# top, bottom, left, right = 41, 40.55, -74.3, -73.6

# #Extracting the Longitude and Latitude of each pickup in our dataset
# Longitudes = df_spatial['Lon'].values
# Latitudes  = df_spatial['Lat'].values

# #Extracting the Longitude and Latitude of each pickup in our reduced dataset
# Longitudes = df_spatial['Lon']
# Latitudes  = df_spatial['Lat']


# plt.figure(figsize=(16, 12))


# plt.ylim(top=top, bottom=bottom)
# plt.xlim(left=left, right=right)


# plt.xlabel('Longitude')
# plt.ylabel('Latitude')
# plt.title('New York Uber Pickups from April to September 2014')
# fig2 , ax1 = plt.subplots()

# ax1 = plt.plot(Longitudes, Latitudes, '.', ms=.8, alpha=.5)

# st.pyplot(fig2)
# CREATING FUNCTION FOR MAPS

def map(d, lat, lon, zoom):
    st.write(pdk.Deck(
        map_style="mapbox://styles/mapbox/light-v9",
        initial_view_state={
            "latitude": lat,
            "longitude": lon,
            "zoom": zoom,
            "pitch": 50,
        },
        layers=[
            pdk.Layer(
                "HexagonLayer",
                data=d,
                get_position=["lon", "lat"],
                radius=100,
                elevation_scale=4,
                elevation_range=[0, 1000],
                pickable=True,
                extruded=True,
            ),
        ]
    ))


def spatial(data1):
	st.title("Airports In Depth")
	# LAYING OUT THE TOP TEMPORAL SECTION OF THE APP
	col_s1,col_s2 = st.beta_columns((2,2))

	with col_s1:
		img=mpimg.imread('index.png')
		st.image(img, caption='New York Uber Pickups from April to September 2014',width=720)


	with col_s2:
		img=mpimg.imread('index1.png')
		st.image(img, caption='New York Uber Pickups from April to September 2014',width=720)


	st.write(
"""Analysing the resultsFrom our spacial visualization we observe that:    

	1. Most of Uber's trips in New York are made from Midtown to Lower Manhattan.
	2. Followed by Upper Manhattan and the Heights of Brooklyn.
	3. Lastly Jersey City and the rest of Brooklyn.

   We see some brighter spots in our heatmap, corresponding to :

	1. LaGuardia Airport in East Elmhurst.
	2. John F. Kennedy International Airport.
	3. Newark Liberty International Airport.

### We know that many airports have specific requirements about where customers can be picked up by vehicles on the Uber platform. We can assume that these three airports have them, since they represent a big part of uber's business in new york""")


#####################################################################################
# INTERACTIVE MAP :
	# LAYING OUT THE TOP SECTION OF THE APP
	col1_1, col1_2 = st.beta_columns((2,3))

	with col1_1:
	    hour_selected = st.slider("Select hour of pickup", 0, 23)


	with col1_2:
	    st.write(
	    """
	    ##
	    Examining how Uber pickups vary over time in New York City's and at its major regional airports.
	    By sliding the slider on the left you can view different slices of time and explore different transportation trends.
	    """)

	# FILTERING DATA BY HOUR SELECTED
	data1 = data1[data1[DATE_TIME].dt.hour == hour_selected]

	# LAYING OUT THE MIDDLE SECTION OF THE APP WITH THE MAPS
	col2_1, col2_2, col2_3, col2_4 = st.beta_columns((2,1,1,1))

	# SETTING THE ZOOM LOCATIONS FOR THE AIRPORTS
	la_guardia= [40.7900, -73.8700]
	jfk = [40.6650, -73.7821]
	newark = [40.7090, -74.1805]
	zoom_level = 12
	midpoint = (np.average(data1["lat"]), np.average(data1["lon"]))

	with col2_1:
	    st.write("**All New York City from %i:00 and %i:00**" % (hour_selected, (hour_selected + 1) % 24))
	    map(data1, midpoint[0], midpoint[1], 11)

	with col2_2:
	    st.write("**La Guardia Airport**")
	    map(data1, la_guardia[0],la_guardia[1], zoom_level)

	with col2_3:
	    st.write("**JFK Airport**")
	    map(data1, jfk[0],jfk[1], zoom_level)

	with col2_4:
	    st.write("**Newark Airport**")
	    map(data1, newark[0],newark[1], zoom_level)

	# FILTERING DATA FOR THE HISTOGRAM
	filtered = data1[
	    (data1[DATE_TIME].dt.hour >= hour_selected) & (data1[DATE_TIME].dt.hour < (hour_selected + 1))
	    ]

	hist = np.histogram(filtered[DATE_TIME].dt.minute, bins=60, range=(0, 60))[0]

	chart_data = pd.DataFrame({"minute": range(60), "pickups": hist})

	# LAYING OUT THE HISTOGRAM SECTION

	st.write("")

	st.write("**Breakdown of rides per minute between %i:00 and %i:00**" % (hour_selected, (hour_selected + 1) % 24))

	st.altair_chart(alt.Chart(chart_data)
	    .mark_area(
	        interpolate='step-after',
	    ).encode(
	        x=alt.X("minute:Q", scale=alt.Scale(nice=False)),
	        y=alt.Y("pickups:Q"),
	        tooltip=['minute', 'pickups']
	    ).configure_mark(
	        opacity=0.5,
	        color='red'
	    ), use_container_width=True)


##############################################

#######################################################
def cluster():
	

	centroids1 = np.array([[ 40.76649043, -73.97209818],
	       [ 40.66538846, -73.76249056],
	       [ 40.68750005, -73.96426561],
	       [ 40.69836122, -74.20257751],
	       [ 40.73136069, -73.99805303],
	       [ 40.7965747 , -73.87295598]])

	names = ["Park Rd, New York, NY 10019, USA" , "182-11 145th Ave, Queens, NY 11413, USA" , "14 Clifton Pl, Brooklyn, NY 11238, USA","Dayton, Newark, NJ 07114, USA","5th Avenue at, Washington Square N, New York, NY 10012, United States","Q4WG+JR New York, USA"]

	cent = pd.DataFrame(data=centroids1,columns=["lat", "lon"])
	# cent.head()

	centroid = cent.values.tolist()
	m = folium.Map(width=500,height=500,location=[40.79658011772687, -73.87341741832425], zoom_start = 9)
	
	for point in range(0, len(centroid)):
		txt = '<i>'+ names[point] +'</i>'
		folium.Marker(centroid[point], popup = txt).add_to(m)
		
	folium_static(m,width=500)



def geo_clustering():

	base()

	st.title("Geo-Spatial Clustering:")

	st.subheader("Identified centroids as their Hubs: 6 HUBS using KMEANS - Elbow method")
	
	# LAYING OUT THE TOP TEMPORAL SECTION OF THE APP
	col_c1,col_c2,col_c3 = st.beta_columns((2,2,2))

	with col_c1:
		img=mpimg.imread('kmeans.png')
		st.image(img, caption='Elbow Method - no of clusters identifies : 6',width=520,height=720)
		
	with col_c2:
		cluster()
	
	with col_c3:
		img=mpimg.imread('index2.png')
		st.image(img, caption='Hubs Covered regions',width=600,height=720)
		

	st.header("Analysis:")
	st.markdown("""	```

1. Uber can use these centroids as their hubs.  \n Whenever Uber received a new ride request, they can check the closeness with each of these centroids. Whichever particular centroid is closer then the Uber can direct the vehicle from that particular location to the customer location.

2. Uber has many drivers and providing services to many locations. \n If Uber knows the hub (particular centroid), and if they are getting a lot of ride request then strategically they can place their driver’s in good location wherein probability of getting a ride request are huge. \n This will help Uber to serve the customer faster as vehicles are placed closer to the location and also it help to grow their business.			

3. Uber can make use of these centroids for the optimal placing of their vehicles. **(Reason for Kmeans Clustering)**  \n They can find which centroid at which part of the day more ride request come in. \n For example, if Uber get more request from centroid 0 (cluster 1) at 11 AM, but very less request from centroid 3 (cluster 4), then they can redirect the vehicles to cluster 1 from cluster 4 (if more vehicle presence in cluster 4).

4. Uber can use these centroids for optimal pricing by analyzing which cluster deals with maximum requests, peak times etc. \n Suppose, if they don’t have too many vehicles to be sent to a particular location (more demand), then they can do optimal pricing as demand is high and supply is less.
				""")
		
	


	# col_c21,col_c22 = st.beta_columns((2,3))

	# with col_c21:
		# st.write("""

		# 	1. Uber can use these centroids as their hubs. Whenever Uber received a new ride request, they can check the closeness with each of these centroids. Whichever particular centroid is closer then the Uber can direct the vehicle from that particular location to the customer location.
		# 	2. Uber has many drivers and providing services to many locations. If Uber knows the hub (particular centroid), and if they are getting a lot of ride request then strategically they can place their driver’s in good location wherein probability of getting a ride request are huge. This will help Uber to serve the customer faster as vehicles are placed closer to the location and also it help to grow their business.
		# 	3. Uber can make use of these centroids for the optimal placing of their vehicles. They can find which centroid at which part of the day more ride request come in. For example, if Uber get more request from centroid 0 (cluster 1) at 11 AM, but very less request from centroid 3 (cluster 4), then they can redirect the vehicles to cluster 1 from cluster 4 (if more vehicle presence in cluster 4).
		# 	4. Uber can use these centroids for optimal pricing by analyzing which cluster deals with maximum requests, peak times etc. Suppose, if they don’t have too many vehicles to be sent to a particular location (more demand), then they can do optimal pricing as demand is high and supply is less.

		# """)


	# with col_c22:
	# 	cluster()
	

# #Setting up the limits
# top, bottom, left, right = 41, 40.55, -74.3, -73.6


# #Extracting the Longitude and Latitude of each pickup in our reduced dataset
# Longitudes_reduced = df['lon']
# Latitudes_reduced  = df['lat']

# from matplotlib.pyplot import figure
# import matplotlib.pyplot as plt
# import matplotlib.cm as cm
# fig3, ax = plt.subplots(figsize=[20, 12])
# centroid_scatter = ax.scatter(centroids1[:, 0],centroids1[:, 1], marker="o", linewidths=2, c='c', s=120)
# facility_scatter = ax.scatter(Longitudes_reduced, Latitudes_reduced, c=algorithm.predict(clus), cmap = cm.Dark2, edgecolor='None', alpha=0.7, s=10)
# plt.ylim(top=top, bottom=bottom)
# plt.xlim(left=left, right=right)
# ax.set_xlabel('Longitude', fontsize=24)
# ax.set_ylabel('Latitude', fontsize = 24)
# st.pyplot(fig3)
# # display(fig)






def vehicle():
	#Number of Active Vehicles By Weekday

	vehicle_data['Date_time'] = pd.to_datetime(vehicle_data['date'])

	#Adding usufull colomns
	vehicle_data['Month'] = vehicle_data['Date_time'].dt.month_name()
	vehicle_data['Weekday'] = vehicle_data['Date_time'].dt.day_name()
	vehicle_data['Day'] = vehicle_data['Date_time'].dt.day
	vehicle_data['Trips_per_vehice'] = vehicle_data['trips']//vehicle_data['active_vehicles']

	container112 = st.beta_container()
	with container112.beta_expander("Data View"):
		st.dataframe(vehicle_data.head())

	
	figv1 , ax_v1 = plt.subplots()
	figv1.set_size_inches(12, 8)
	ax_v1 = sns.barplot(x='Weekday', y='active_vehicles', data=vehicle_data)

	figv2 , ax_v2 = plt.subplots()
	figv2.set_size_inches(12, 8)
	ax_v2 = sns.boxplot(x='Trips_per_vehice', y='Weekday',  data=vehicle_data)
	
	

	figv3 , ax_v3 = plt.subplots()
	figv3.set_size_inches(12, 8)
	ax_v3 = sns.boxplot(x='trips', y='Weekday',  data=vehicle_data)
	
	

	colv1,colv2,colv3 = st.beta_columns((2,2,2))


	with colv1:
		container2 = st.beta_container()
		container2.subheader("Active Vehicles by Weekday")
		container2.pyplot(figv1,width=6)

	with colv2:
		containere = st.beta_container()
		containere.subheader("No of Trips by Weekday")
		containere.pyplot(figv3)	
		
		
	with colv3:
		containere = st.beta_container()
		containere.subheader("No of Trips per Vehicle")
		containere.pyplot(figv2)
		


	

	colm1,colm2,colm3 = st.beta_columns((2,2,2))

	with colm1:
		st.write(" ")
	with colm2:
		with st.beta_expander("Key Points:"):

			st.write("* Thursday, Friday and Saturday had more number of active Vehicles")
			st.write("* Around 9 trips per vehicle is the most common **(Target should be: Atleast Each vehicle do atleast 9 trips)**")
			st.write("* There are vehicles which have run even more than 10 trips per day ")

	with colm3:
		with st.beta_expander("Derived Attribute:"):
			st.write(""" No of Trips per Vehicle is a Derived attribute where:
					Number_of_trips per Vehicle = (No of Trips / No of Active_Vehicles""")
			st.write("""Since Both Trips & Active_Vehicles show a +ve Correlation with Week day  , thus trips_per_vehicle attribute also shows a +ve correlation""")		 	


def Business():
	colb1,colb2 = st.beta_columns((2,2))


	with colb1:
		img=mpimg.imread('UberVsother.png')
		st.image(img, caption='Uber Business vs other businesses',width=720)

	with colb2:
		img1=mpimg.imread('growth.png')
		st.image(img1,width=720)
		
	colm1,colm2,colm3 = st.beta_columns((2,2,2))

	with colm1:
		st.write(" ")
	with colm2:
		with st.beta_expander("Key Points:"):
			st.write("* From June of 2014 , Uber started to see some Competition in the market")
			st.write("* Lyft Cabs & Skyline Taxis , showed promising growth")
			st.write("* Although Uber did not seem to hurt the business of other companies in 2014 as all other companies experienced a growth in their business along with Uber")
			st.write("* Since Uber operated @ a very large scale compared to its competitors, its business flurished as usual.")
	
	with colm3:
		st.write(" ")

	st.subheader("Uber Business in 2015 : Uber rides continued to grow in the first half of 2015 as well")	
	colb1,colb2 = st.beta_columns((2,2))

	with colb1:
		img=mpimg.imread('monthly.png')
		st.image(img, caption='',width=720)

	with colb2:
		img1=mpimg.imread('gwothmonth.png')
		st.image(img1,width=720)

	colmb1,colmb2,colmb3 = st.beta_columns((2,2,2))

	with colmb1:
		st.write(" ")
	with colmb2:
		with st.beta_expander("Key Points:"):
			st.write("* Uber also did well in Each month of 2015 : Jan to Jun (The next year)")
			st.write("* February to April saw a stagnant gowth , but following months its customer base increased.")
	with colmb3:
		st.write(" ")
	
	st.subheader("Uber Vs Lyft: Operating Regions")	
	colu1,colu2 = st.beta_columns((2,2))

	with colu1:
		img=mpimg.imread('index.png')
		st.image(img, caption='Uber Pickup Regions',width=720)

	with colu2:
		img1=mpimg.imread('Lyft_heat.png')
		st.image(img1,caption='Lyft Pickup Regions',width=720)


	colmu1,colmu2,colmu3 = st.beta_columns((2,2,2))
	
	with colmu1:
		st.write(" ")
	with colmu2:
		with st.beta_expander("Key Points:"):
			st.write("* Uber has covered almost every part of the NewYork City.")
			st.write("* While Lyft Covers the major Boroughs of the NewYork City , maybe due to its operational size ,its not able to cover the Airport Regions and misses out completely on Newark Region.")
	with colmu3:
		st.write(" ")

		
		


st.title("Cab-share analysis using Spatio-temporal analysis and Geospatial Clustering")
st.subheader("Select View For the DashBoard")
option = st.selectbox('',('Temporal','Airports In Depth','Base Station & Identified Hubs','Business Analysis of Competition','Vehicle analysis'))

if(option=="Temporal"):
	temporal()

elif(option=="Airports In Depth"):
	spatial(data1)

elif(option=="Base Station & Identified Hubs"):
	geo_clustering()

elif(option=="Business Analysis of Competition"):
	Business()

elif(option=="Vehicle analysis"):
	vehicle()
