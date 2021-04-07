#!/usr/bin/env python
# coding: utf-8

# # Thermal Comfort Predictor - Lower
# ## Developed by Alex C. & Jordan V. @ Ryerson's Capstone 2021 </span>
# ### [Front-End Website](https://share.streamlit.io/vanrjord/occupantthermalcomfort/main/comfort.py) 
# ---
# #### Note: Program can be retrained with any dataset
# #### Email: q1cheng@ryerson.ca or jvanriel@ryerson.ca for any questions regarding this program

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score
from sklearn.svm import SVC
from sklearn import metrics

# the error metric (ROC/AUC)
from sklearn.metrics import roc_auc_score
import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import streamlit as st
import webbrowser

df = pd.read_csv('Lower_Data.csv')
url = 'https://www.weatherstats.ca/' # URL for weather website


# # Check if there's missing data
# if there's missing data, need to address it

# In[2]:


# False = no missing data, True = Missing data aka NaN
df.isnull().any()


# # Split Data
# get y values (Hot , Cold)
X = df
y = X.pop("Classification")

#look at data
X.describe()
y.describe()

data = list(X.columns.values)

# 80(train)/20(test)
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, test_size=0.2)

# # Optimization Finish
# ## Final Model

# # Evaluate Model
# Streamlit Interface

# Header
st.write("""
# Thermal Comfort Program
### Predict thermal comfort given variables on the left """)

# Sidebar
st.sidebar.header('User Input Parameters')
col1, col2, col3 = st.sidebar.beta_columns(3)

# Creating the dataset that the model will make the prediction from
# uses sliders to choose values

# Ex: upper_north_sp = st.sidebar.slider(Title, Min Value, Max Value, Current Value)
def user_input_features(): 
    st.sidebar.subheader("Internal Factors")
    upper_north_sp = st.sidebar.slider('North Upper Setpoint', 90.0, 170.8531, 142.427 )
    st.markdown('<style>#vg-tooltip-element{z-index: 1000051}</style>',
            unsafe_allow_html=True)
    upper_west_sp = st.sidebar.slider('West Upper Setpoint', 90.0, 161.5931, 123.164)
    upper_south_sp = st.sidebar.slider('South Upper Setpoint', 90.0, 155.0126, 128.0942)
    upper_east_sp = st.sidebar.slider('East Upper Setpoint', 80.0, 161.5931, 143.7969)
    shroud_north = st.sidebar.slider('Solar Shroud North', 3.547888, 67.83803, 38.52079)
    shroud_west = st.sidebar.slider('Solar Shroud West', 1.27017, 91.37299, 39.32901)
    shroud_south = st.sidebar.slider('Solar Shroud South', 6.603269, 83.13541, 51.18748)
    shroud_east = st.sidebar.slider('Solar Shroud East', 2.49277, 83.70346, 50.12392)
    shroud_north_pst1hr = st.sidebar.slider('Lag Solar Shroud North', 4.054758, 70.46021, 37.3063)
    shroud_west_pst1hr = st.sidebar.slider('Lag Solar Shroud West', 0.7772345, 88.07684, 37.72742)
    shroud_south_pst1hr = st.sidebar.slider('Lag Solar Shroud South', -0.4071831, 84.75342, 43.22388)
    shroud_east_pst1hr = st.sidebar.slider('Lag Solar Shroud East', 1.824043, 73.93447, 52.22408)
    upper_west_zone = st.sidebar.slider('Upper West Zone Temp', 64.31408, 75.46066, 73.15348)
    upper_north_zone = st.sidebar.slider('Upper North Zone Temp', 64.81843, 73.16819, 70.84258)
    upper_east_zone = st.sidebar.slider('Upper East Zone Temp', 66.42004, 75.45654, 71.93165)
    upper_south_zone = st.sidebar.slider('Upper South Zone Temp', 66.58105, 75.50656, 71.6673)
    OAT = st.sidebar.slider('Outside Air Temp', 2.433141, 69.12512, 41.30745)
    OAT_pst1hr = st.sidebar.slider('Lagging Outside Air Temp', 3.377104, 70.07122, 38.36398)
    st.sidebar.subheader("External Factors")
    if st.sidebar.button("Open Weather Data Website","https://www.weatherstats.ca/" ):
        webbrowser.open_new_tab(url)
    pressure_station = st.sidebar.slider('pressure_station', 98.62, 103.66, 100.42)
    wind_speed = st.sidebar.slider('wind_speed', 0.0, 55.0, 9.0)
    relative_humidity = st.sidebar.slider('relative_humidity', 31.0 , 100.0, 83.0)
    max_air_temp_pst1hr = st.sidebar.slider('max_air_temp_pst1hr', 2.48, 71.6, 37.22)
    min_air_temp_pst1hr = st.sidebar.slider('min_air_temp_pst1hr', 0.86, 67.82, 35.42)
    data1 = {'upper_north_sp': upper_north_sp,
            'upper_west_sp': upper_west_sp,
            'upper_south_sp': upper_south_sp,
            'upper_east_sp': upper_east_sp,
            'shroud_north': shroud_north,
            'shroud_west': shroud_west,
            'shroud_south': shroud_south,
            'shroud_east': shroud_east,
            'shroud_north_pst1hr': shroud_north_pst1hr,
            'shroud_west_pst1hr': shroud_west_pst1hr,
            'shroud_south_pst1hr': shroud_south_pst1hr,
            'shroud_east_pst1hr': shroud_east_pst1hr,
            'upper_west_zone': upper_west_zone,
            'upper_north_zone': upper_north_zone,
            'upper_east_zone': upper_east_zone,
            'upper_south_zone': upper_south_zone,
            'OAT': OAT,
            'OAT_pst1hr': OAT_pst1hr,
            'pressure_station': pressure_station,
            'wind_speed': wind_speed,
            'relative_humidity': relative_humidity,
            'max_air_temp_pst1hr': max_air_temp_pst1hr,
            'min_air_temp_pst1hr': min_air_temp_pst1hr,
            }
    features = pd.DataFrame(data1, index=[1064])
    return features
df1 = user_input_features()

def model(df1): #creating a function that can be called to run the model
    st.subheader('User Input parameters')
    st.write(df1)

    # the model
    best_model = RandomForestClassifier(max_depth=None, n_estimators=1000, oob_score=True, n_jobs=-1, random_state=42, max_features="auto", min_samples_leaf=2)

    best_model.fit(X_train, y_train)

    prediction = best_model.predict(df1)
    prediction_proba = best_model.predict_proba(df1)
    classes = best_model.classes_

    # Displaying a subheader for the classes (Hot & COLD)
    st.subheader('Class labels and their corresponding index number')
    st.write(classes)

    # Displaying the Prediction
    st.subheader('Prediction')
    st.write(prediction)
    
    # Displaying the Prediction probability
    st.subheader('Prediction Probability')
    st.write(prediction_proba)

    chart_data = pd.DataFrame(
            prediction_proba,
            columns=["Cold", "Hot"])

    st.bar_chart(chart_data)

    classification = []


    #Exporting to csv or SQL server
    Live_Data_Lower = pd.read_csv('Live_Data_Lower.csv')
    Live_Data_Lower = pd.DataFrame(Live_Data_Lower)
    length = len(Live_Data_Lower.index)
    length
    output_data = Live_Data_Lower.copy()
    def column(matrix, i):
        return[row[i] for row in matrix]

    cold_column = column(prediction_proba, 0)
    hot_column = column(prediction_proba, 1)

    print(classes[0], cold_column)
    print(classes[1], hot_column)

    def compare():
        for i in range (0, length):
            if cold_column[i] < hot_column[i]:
                classification.insert(i, "HOT")
            else:
                classification.insert(i, "COLD")
        return[classification]

    output = compare()
    output = np.array(output)
    output = np.transpose(output)

     # Columns that will be included in the outputted dataset
    output_data["Predictions - COLD or HOT"] = prediction
    output_data["Predictions - Probability of Hot"] = hot_column
    output_data["Predictions - Probability of Cold"] = cold_column
    # output_data
    output_data.to_csv("Thermal_Comfort_Prediction_Lower.csv", sep = '\t', index = False)

# Create a button, that when clicked, runs the program
if(st.button("Run")):
    st.text("Running")
    model(df1)


