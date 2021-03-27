#!/usr/bin/env python
# coding: utf-8

# # Import Model and Dataset

# In[1]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score
from sklearn.svm import SVC
from sklearn import metrics
import streamlit as st

# the error metric (ROC/AUC)
from sklearn.metrics import roc_auc_score
import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# to look at all columns in dataset if needed
def printall(X, max_rows=10):
    from IPython.display import display, HTML
    display(HTML(X.to_html(max_rows=max_rows)))

df = pd.read_csv('upperdata.csv')


# # Check if there's missing data
# if there's missing data, need to address it

# In[2]:


# False = no missing data, True = Missing data aka NaN
df.isnull().any()


# # Split Data

# In[3]:


# get y values (Hot , Cold)
X = df
y = X.pop("Classification")

#look at data
X.describe()


# In[4]:


y.describe()


# # Scaling Input Data

# In[5]:


data = list(X.columns.values)
#for i in range (0, len(X.columns)):
 #   print(data[i:i+1])
# data


# In[6]:


# scale_vars = ['unorth', 'uwest', 'usouth', 'ueast', 'Shroud North', 'Shroud West', 'Shroud South', 'Shroud East', 'Lag Shroud North', 'Lag Shroud West', 'Lag Shroud South', 'LagShroud East', 'uWEST zone', 'uNORTH zone', 'uEAST zone', 'uSOUTH zone', 'OAT', 'Lagging OAT', 'pressure_station', 'wind_speed', 'relative_humidity', 'max_air_temp_pst1hr', 'min_air_temp_pst1hr']
# scaler = MinMaxScaler()
# X[scale_vars] = scaler.fit_transform(X[scale_vars])
# X.head()


# # Split Dataset into Testing and Training

# In[7]:


# 80(train)/20(test)
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, test_size=0.2)


# # Random Forest Classifier

# In[8]:


# Call randomforest with no parameter change == no optimization
# model = RandomForestClassifier()
# train model
# model.fit(X_train, y_train)


# # Accuracy Score of Training and Testing model w/o Optimization
# Quick way to see accuracy. This is meaningless because it can very well be overfitting the data.

# In[9]:


# print("Training Accuracy is: ", model.score(X_train, y_train))
# print("Testing Accuracy is: ", model.score(X_test, y_test))


# # Start Optimization

# In[10]:


# Base model
# model = RandomForestClassifier(n_estimators=100, oob_score=True, n_jobs=-1, random_state=42)
# fit using training data
# model.fit(X_train, y_train)


# ## Out of Bag (oob score) more accurate way of measuring accuracy 

# In[11]:


# oob_score = model.oob_score_
# print("The oob score is:", oob_score)


# ### Benchmark

# In[12]:


# pred_train = model.oob_decision_function_[:,1]
# print("C-stat: ", roc_auc_score(y_train, pred_train))


# # Optimize
# carry on results from each phase

# ### n_estimators
# number of trees in forest

# In[13]:


# results = []
# n_estimator_options = [30, 50, 100, 200, 500, 1000, 2000, 4000, 6000]

# for trees in n_estimator_options:
#     model = RandomForestClassifier(trees, oob_score=True, n_jobs=-1, random_state=42)
#     model.fit(X_train,y_train)
#     print(trees, "trees")
#     roc = roc_auc_score(y_train, model.oob_decision_function_[:,1])
#     print("c-stat: ", roc)
#     results.append(roc)
#     print("")

# pd.Series(results, n_estimator_options).plot()    


# pick the number of trees at max point, more tree = longer time to process

# ### max_features

# In[14]:


# results = []
# max_features_options = ["auto", None, "sqrt", "log2", 0.9, 0.2]

# for max_features in max_features_options:
#     model = RandomForestClassifier(n_estimators=2000, oob_score=True, n_jobs=-1, random_state=42, max_features=max_features)
#     model.fit(X_train,y_train)
#     print(max_features, "option")
#     roc = roc_auc_score(y_train, model.oob_decision_function_[:,1])
#     print("C-stat", roc)
#     results.append(roc)
#     print("")

# pd.Series(results, max_features_options).plot(kind="barh")


# ## min_sample_leaf
# 

# In[15]:


# results = []
# min_samples_leaf_options = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

# for min_samples in min_samples_leaf_options:
#     model = RandomForestClassifier(n_estimators=2000, oob_score=True, n_jobs=-1, random_state=42, max_features="auto", min_samples_leaf=min_samples)
#     model.fit(X_train,y_train)
#     print(min_samples, "minimum samples")
#     roc = roc_auc_score(y_train, model.oob_decision_function_[:,1])
#     print("C-stat", roc)
#     results.append(roc)
#     print("")

# pd.Series(results, min_samples_leaf_options).plot()         


# ## max_depth

# In[16]:


# results = []
# max_depth_options = [None, 2, 4, 6, 8, 10, 12, 14, 16]

# for max_depth in max_depth_options:
#     model = RandomForestClassifier(max_depth=max_depth, n_estimators=2000, oob_score=True, n_jobs=-1, random_state=42, max_features="auto", min_samples_leaf=5)
#     model.fit(X_train, y_train)
#     print(max_depth, "max depth")
#     roc = roc_auc_score(y_train, model.oob_decision_function_[:,1])
#     print("C-stat", roc)
#     results.append(roc)
#     print("")

# pd.Series(results, max_depth_options).plot()


# pick value at peak point

# # Optimization Finish
# ## Final Model

# In[17]:


# best_model = RandomForestClassifier(max_depth=None, n_estimators=2000, oob_score=True, n_jobs=-1, random_state=42, max_features="auto", min_samples_leaf=5)


# # Evaluate Model

# In[18]:
st.write("""
# Thermal Comfort Program
### Predict thermal comfort given variables on the left """)

st.sidebar.header('User Input Parameters')
# ['unorth', 'uwest', 'usouth', 'ueast', 'Shroud North', 'Shroud West', 'Shroud South', 'Shroud East', 'Lag Shroud North', 'Lag Shroud West', 'Lag Shroud South', 'LagShroud East', 'uWEST zone', 'uNORTH zone', 'uEAST zone', 'uSOUTH zone', 'OAT', 'Lagging OAT (1hr)', 'pressure_station', 'wind_speed', 'relative_humidity', 'max_air_temp_pst1hr', 'min_air_temp_pst1hr']
def user_input_features():
    unorth = st.sidebar.slider('unorth', 90.0, 163.9, 123.2644)
    uwest = st.sidebar.slider('uwest', 90.0, 162.289, 116.8894)
    usouth = st.sidebar.slider('usouth', 90.0, 162.289, 116.1613)
    ueast = st.sidebar.slider('ueast', 90.0, 162.289, 112.211)
    Shroud_North = st.sidebar.slider('Shroud North', 0.32574, 1.0, 0.621799921)
    Shroud_West = st.sidebar.slider('Shroud West', -0.40503, 1.0, 0.64936969)
    Shroud_South = st.sidebar.slider('Shroud South', 0.216652, 1.0, 0.903140685)
    Shroud_East = st.sidebar.slider('Shroud East', 0.059244, 1.0, 1.0)
    Lag_Shroud_North = st.sidebar.slider('Lag Shroud North', 0.300682, 1.0, 0.738706428)
    Lag_Shroud_West = st.sidebar.slider('Lag Shroud West', 0.30068204, 1.0, 0.714224336)
    Lag_Shroud_South = st.sidebar.slider('Lag Shroud South', -0.160854969, 1.0, 0.9965169)
    LagShroud_East = st.sidebar.slider('LagShroud East', 0.039509997, 1.0, 1.0)
    uWEST_zone = st.sidebar.slider('uWEST zone', 65.58956, 75.75271, 71.93018)
    uNORTH_zone = st.sidebar.slider('uNORTH zone', 64.38702, 72.99919, 71.89948)
    uEAST_zone = st.sidebar.slider('uEAST zone', 65.54574, 76.11459, 72.59882)
    uSOUTH_zone = st.sidebar.slider('uSOUTH zone', 65.47312, 77.03263, 73.57618)
    OAT = st.sidebar.slider('OAT', -0.4084013, 78.90985, 25.48777)
    Lagging_OAT = st.sidebar.slider('Lagging OAT', 0.6155012, 77.03983, 24.35927)
    pressure_station = st.sidebar.slider('pressure_station', 98.09, 103.66, 99.59)
    wind_speed = st.sidebar.slider('wind_speed', 0.0, 61.0, 50.0)
    relative_humidity = st.sidebar.slider('relative_humidity', 31.0 , 100.0, 65.0)
    max_air_temp_pst1hr = st.sidebar.slider('max_air_temp_pst1hr', -0.04, 75.56, 25.88)
    min_air_temp_pst1hr = st.sidebar.slider('min_air_temp_pst1hr', -1.3, 72.32, 24.8)
    data1 = {'unorth': unorth,
            'uwest': uwest,
            'usouth': usouth,
            'ueast': ueast,
            'Shroud_North': Shroud_North,
            'Shroud_West': Shroud_West,
            'Shroud_South': Shroud_South,
            'Shroud_East': Shroud_East,
            'Lag_Shroud_North': Lag_Shroud_North,
            'Lag_Shroud_West': Lag_Shroud_West,
            'Lag_Shroud_South': Lag_Shroud_South,
            'LagShroud_East': LagShroud_East,
            'uWEST_zone': uWEST_zone,
            'uNORTH_zone': uNORTH_zone,
            'uEAST_zone': uEAST_zone,
            'uSOUTH_zone': uSOUTH_zone,
            'OAT': OAT,
            'Lagging_OAT': Lagging_OAT,
            'pressure_station': pressure_station,
            'wind_speed': wind_speed,
            'relative_humidity': relative_humidity,
            'max_air_temp_pst1hr': max_air_temp_pst1hr,
            'min_air_temp_pst1hr': min_air_temp_pst1hr,
            }
    features = pd.DataFrame(data1, index=[1064])
    return features

df1 = user_input_features()

st.subheader('User Input parameters')
st.write(df1)
best_model = RandomForestClassifier(max_depth=None, n_estimators=2000, oob_score=True, n_jobs=-1, random_state=42, max_features="auto", min_samples_leaf=5)

best_model.fit(X_train, y_train)

prediction = best_model.predict(df1)
prediction_proba = best_model.predict_proba(df1)
classes = best_model.classes_


st.subheader('Class labels and their corresponding index number')
st.write(classes)

st.subheader('Prediction')
st.write(prediction)
#st.write(prediction)

st.subheader('Prediction Probability')
st.write(prediction_proba)

chart_data = pd.DataFrame(
        prediction_proba,
        columns=["Cold", "Hot"])

st.bar_chart(chart_data)

classification = []


#Exporting to csv or SQL server
unseen_data = pd.read_csv('unseen.csv')
unseen_data = pd.DataFrame(unseen_data)
length = len(unseen_data.index)
length
output_data = unseen_data.copy()
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

output_data["Predictions - COLD or HOT"] = prediction
output_data["Predictions - Probability of Hot"] = hot_column
output_data["Predictions - Probability of Cold"] = cold_column
output_data
output_data.to_csv("Thermal_Comfort_Prediction_Upper.csv", sep = '\t', index = False)

# oob_score = model.oob_score_
# print("The final oob score is:", oob_score)


# expect only 0-2% percent change from non-optimized

# In[19]:


# pred_train = best_model.oob_decision_function_[:,1]
# print("The final C-stat: ", roc_auc_score(y_train, pred_train))


# In[20]:


# metrics.plot_roc_curve(best_model, X_test, y_test) 
# plt.show() 


# In[21]:


# y_predictions = best_model.predict(X_test)
# score = accuracy_score(y_test, y_predictions)
# print("Accuracy score:", score*100)


# In[22]:


# classes = best_model.classes_
# pred_prob = best_model.predict_proba(X_test)
# print("Probability of the first test is")
# for i in range (0, 2):
#    print(classes[i], "is", pred_prob[0][i]*100, "%")


# # Variable Importance Measures

# In[23]:


# Shows which has most importance
# feature_importances = pd.Series(best_model.feature_importances_, index=X.columns)
# feature_importances = feature_importances.sort_values()
# feature_importances.plot(kind="barh", figsize=(7,6))


# # Deploying Model on New Unseen Data

# In[24]:


#Import data from csv or SQL server
# unseen_data = pd.read_csv('yes.csv')
# unseen_data = pd.DataFrame(unseen_data)
#Number of rows -> returns a number that is one less than the number of rows in the datasheet
# length = len(unseen_data.index)
# length


# In[25]:


#Scaling
# df = pd.read_csv('upperdata.csv')
# hello = df
# hello.pop("Classification")
# hello


# # Combine original dataframe with targeted dataframe
# ### Done to normalize targeted data against the original dataframe

# In[26]:


#combine the two dataframes
# hello = pd.DataFrame(hello)
# list(unseen_data.columns.values)
# new_data = pd.concat([unseen_data, hello], ignore_index=True)
# new_data


# In[27]:


# scale_vars2 = ['unorth', 'uwest', 'usouth', 'ueast', 'Shroud North', 'Shroud West', 'Shroud South', 'Shroud East', 'Lag Shroud North', 'Lag Shroud West', 'Lag Shroud South', 'LagShroud East', 'uWEST zone', 'uNORTH zone', 'uEAST zone', 'uSOUTH zone', 'OAT', 'Lagging OAT', 'pressure_station', 'wind_speed', 'relative_humidity', 'max_air_temp_pst1hr', 'min_air_temp_pst1hr']
# scaler = MinMaxScaler()
# hello[scale_vars2] = scaler.fit_transform(hello[scale_vars2])
# hello.head()


# In[28]:


# i = length
# newnew_data = hello[:i]
# newnew_data


# # Making Predictions

# In[29]:


#Making Predictions
# pred_prob = best_model.predict_proba(newnew_data)
# classes = best_model.classes_
# print(classes)
# pred_prob


# In[30]:


#Exporting results as indiviual columns
# def column(matrix, i):
#     return[row[i] for row in matrix]

# cold_column = column(pred_prob, 0)
# hot_column = column(pred_prob, 1)

# print(classes[0], cold_column)
# print(classes[1], hot_column)


# In[31]:


# classification = []

# def compare():
#     for i in range (0, length):
#         if cold_column[i] < hot_column[i]:
#             classification.insert(i, "HOT")
#         else:
#             classification.insert(i, "COLD")
#     return[classification]

# output = compare()
# output = np.array(output)
# output = np.transpose(output)
#output = column(output, 1)
# print(output)


# # Final Result

# In[32]:


# output_data = unseen_data.copy()
# output_data["Predictions - COLD or HOT"] = output
# output_data["Predictions - Probability of Hot"] = hot_column
# output_data["Predictions - Probability of Cold"] = cold_column
# output_data


# In[33]:


#Exporting to csv or SQL server
# output_data.to_csv("Thermal_Comfort_Prediction_Upper", sep = '\t', index = False)


# In[ ]:




