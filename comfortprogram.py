#!/usr/bin/env python
# coding: utf-8

# # Thermal Comfort Predictor - Upper
# ## Developed by Alex C. & Jordan V. @ Ryerson's Capstone 2021 </span>
# ### [Front-End Website](https://share.streamlit.io/vanrjord/occupantthermalcomfort/main/comfortprogram.py) 
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
import time

df = pd.read_csv('Upper_Data.csv')

url = 'https://www.weatherstats.ca/' # URL for weather website

# get y values (Hot , Cold)
X = df
y = X.pop("Classification")

#look at data
X.describe()
y.describe()

data = list(X.columns.values)
# # Split Dataset into Testing and Training
# 80(train)/20(test)
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, test_size=0.2)

# Final Model
# Sidebar
st.sidebar.header('User Input Parameters')
# Creating the dataset that the model will make the prediction from
# uses sliders to choose values

# Ex: upper_north_sp = st.sidebar.slider(Title, Min Value, Max Value, Current Value)
def user_input_features():
    st.sidebar.subheader("Internal Factors")
    upper_north_sp = st.sidebar.slider('North Upper Setpoint', 90.0, 163.9, 123.2644)
    upper_west_sp = st.sidebar.slider('West Upper Setpoint', 90.0, 162.289, 116.8894)
    upper_south_sp = st.sidebar.slider('South Upper Setpoint', 90.0, 162.289, 116.1613)
    upper_east_sp = st.sidebar.slider('East Upper Setpoint', 90.0, 162.289, 112.211)
    shroud_north = st.sidebar.slider('Solar Shroud North', 3.927225, 71.3776, 23.28826)
    shroud_west = st.sidebar.slider('Solar Shroud West', -2.908249, 91.37299, 24.32083)
    shroud_south = st.sidebar.slider('Solar Shroud South',  3.616019, 94.3151, 33.82531)
    shroud_east = st.sidebar.slider('Solar Shroud East', 4.039497, 89.44633, 37.45298)
    shroud_north_pst1hr = st.sidebar.slider('Lag Solar Shroud North', 3.860796, 70.17285, 22.3684)
    shroud_west_pst1hr = st.sidebar.slider('Lag Solar Shroud West', -2.609743, 88.07684, 21.62707)
    shroud_south_pst1hr = st.sidebar.slider('Lag Solar Shroud South', -0.8090598, 89.23934, 30.17503)
    shroud_east_pst1hr = st.sidebar.slider('Lag Solar Shroud East', 2.049355, 90.45381, 30.2805)
    upper_west_zone = st.sidebar.slider('Upper West Zone Temp', 65.58956, 75.75271, 71.93018)
    upper_north_zone = st.sidebar.slider('Upper North Zone Temp', 64.38702, 72.99919, 71.89948)
    upper_east_zone = st.sidebar.slider('Upper East Zone Temp', 65.54574, 76.11459, 72.59882)
    upper_south_zone = st.sidebar.slider('Upper South Zone Temp', 65.47312, 77.03263, 73.57618)
    OAT = st.sidebar.slider('Outside Air Temp', -0.4084013, 78.90985, 25.48777)
    OAT_pst1hr = st.sidebar.slider('Lagging Outside Air Temp', 0.6155012, 77.03983, 24.35927)
    st.sidebar.subheader("External Factors")
    if st.sidebar.button("Open Weather Data Website","https://www.weatherstats.ca/" ):
        webbrowser.open_new_tab(url)
    pressure_station = st.sidebar.slider('pressure_station', 98.09, 103.66, 99.59)
    wind_speed = st.sidebar.slider('wind_speed', 0.0, 61.0, 50.0)
    relative_humidity = st.sidebar.slider('relative_humidity', 31.0 , 100.0, 65.0)
    max_air_temp_pst1hr = st.sidebar.slider('max_air_temp_pst1hr', -0.04, 75.56, 25.88)
    min_air_temp_pst1hr = st.sidebar.slider('min_air_temp_pst1hr', -1.3, 72.32, 24.8)
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


def model_upper(df1):  #creating a function that can be called to run the model
        
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


    #Exporting to csv or SQL server
    Live_Data_Upper = pd.read_csv('Live_Data_Upper.csv')
    Live_Data_Upper = pd.DataFrame(Live_Data_Upper)
    length = len(Live_Data_Upper.index)

    output_data = Live_Data_Upper.copy()
    def column(matrix, i):
        return[row[i] for row in matrix]

    cold_column = column(prediction_proba, 0)
    hot_column = column(prediction_proba, 1)

    print(classes[0], cold_column)
    print(classes[1], hot_column)


    classification = []

    def compare():
        for i in range (0, length):
            if cold_column[i] < hot_column[i]:
                classification.insert(i, "HOT")
            elif (abs(cold_column[i]-hot_column[i]) < 0.2):
                classification.insert(i, "NEUTRAL")
            else:
                classification.insert(i, "COLD")
        return[classification]

    output = compare()
    output = np.array(output)
    output = np.transpose(output)

    # Columns that will be included in the outputted dataset
    output_data = Live_Data_Upper.copy()
    output_data["Predictions - COLD or HOT"] = prediction
    output_data["Predictions - Probability of Hot"] = hot_column
    output_data["Predictions - Probability of Cold"] = cold_column
    #output_data
    output_data.to_csv("Thermal_Comfort_Prediction_Upper.csv", sep = '\t', index = False)
    st.success('Done!')

def model_lower(df1): #creating a function that can be called to run the model
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
    #length
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
    st.success('Done!')
# Streamlit Interface
# Header
st.title("Occupant Thermal Comfort Program")
st.header("How to use?")
st.info("\n 1. Find the Parameter you would like to manipulate from the sidebar \n 2. Adjust the Slider to the appropriate value \n 3. Once Satisfied, Click the 'Run' Button")
col1, col2 = st.beta_columns([5,5])
with col1:
     if(st.button("Run for Upper")): # Create a button, that when clicked, runs the program
             
             model_upper(df1)   
with col2:
     if(st.button("Run for Lower")): # Create a button, that when clicked, runs the program
             
             model_lower(df1)   




