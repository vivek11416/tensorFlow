# -*- coding: utf-8 -*-
"""
Created on Thu May  7 12:32:43 2020

@author: vivek
"""


import pandas as pd
import  numpy as np
import tensorflow as tf
from sklearn import preprocessing


dataset = pd.read_csv("cal_housing_clean.csv")

#==============data into x and Y===========

x = dataset.iloc[:,:-1]
y = dataset['medianHouseValue']

#=================Splitting the data===========
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=101)

#=============normalizing the data==========

cols_to_norm = ['housingMedianAge','totalRooms','totalBedrooms','population','households','medianIncome']

x_train[::] = preprocessing.MinMaxScaler().fit_transform(x_train[::])
x_test[::] = preprocessing.MinMaxScaler().fit_transform(x_test[::])


x_train = pd.DataFrame(data=x_train,columns = x_train.columns,index = x_train.index)
x_test = pd.DataFrame(data=x_test,columns = x_test.columns,index = x_test.index)



#======================== create feature columns=======================
feat_cols = []
for columns in x.columns:
    varName = "var"+columns 
    varName = tf.feature_column.numeric_column(columns)
    feat_cols.append(varName)
     
#================Input functio for estimator object =================

input_func = tf.estimator.inputs.pandas_input_fn(x=x_train , y =y_train, batch_size = 10 , num_epochs=1000 ,shuffle = True)

model = tf.estimator.DNNRegressor(hidden_units=[6,6,6], feature_columns= feat_cols)

model.train(input_func,steps=100000)

pred_input_func =  tf.estimator.inputs.pandas_input_fn(x=x_test,
                                                       batch_size=10,
                                                       num_epochs=1,
                                                       shuffle=False)

predicted_values_gen = model.predict(pred_input_func)

predictions = list(predicted_values_gen)

predictions


final_preds = []

for pred in predictions:
    final_preds.append(pred['predictions'])
    
    
    
from sklearn.metrics import mean_squared_error

mean_squared_error(y_test,final_preds)**0.5
