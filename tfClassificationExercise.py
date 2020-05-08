# -*- coding: utf-8 -*-
"""
Created on Thu May  7 16:58:01 2020

@author: vivek
"""


import pandas as pd
import  numpy as np
import tensorflow as tf
from sklearn import preprocessing


census = pd.read_csv("census_data.csv")
census['income_bracket'].unique()


def label_fix(label):
    if label == census['income_bracket'].unique()[0]:
        return 0
    else:
        return 1
    

census['income_bracket'] = census['income_bracket'].apply(label_fix)


x= census.drop('income_bracket',axis = 1)

y= census['income_bracket']

from sklearn.model_selection import train_test_split


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

census.columns

feat_cols = []
categorical_varibles = ['workclass', 'education', 'marital_status','occupation', 'relationship', 'gender','native_country']

for category in categorical_varibles:
    varName = category
    
    varName = tf.feature_column.categorical_column_with_hash_bucket(category, hash_bucket_size=1000)
    feat_cols.append(varName)
    
numerical_varibles = ['age','education_num','capital_gain','capital_loss', 'hours_per_week']

for numCategory in numerical_varibles:
    varName = numCategory
    
    varName = tf.feature_column.numeric_column(numCategory)
    feat_cols.append(varName)
    


input_func = tf.estimator.inputs.pandas_input_fn(x=x_train,y=y_train,batch_size=100,num_epochs=None,shuffle=True)

model = tf.estimator.LinearClassifier(feature_columns=feat_cols)

model.train(input_func,steps = 10000)


pred_fn = tf.estimator.inputs.pandas_input_fn(x=x_test,batch_size=len(x_test),shuffle = False)

predictions = list(model.predict(pred_fn))

predictions

final_pred = []

for preds in predictions:
    final_pred.append(preds['class_ids'][0])
    
    
from sklearn.metrics import classification_report

print(classification_report(y_test, final_pred))
    