import keras.layers
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from keras.layers import Dense
import seaborn as sns


#mpg= miles per gallon

column_names = ['mpg','cylinders','displacement','horsepower','weight','acceleration','model_year','origin']

data = pd.read_csv('data/auto+mpg/auto-mpg.data',names=column_names,na_values='?',comment='\t',sep=' ',skipinitialspace=True)

data = data.drop('origin',axis=1)
data = data.dropna()

train_dataset = data.sample(frac=0.8,random_state=0)
test_dataset = data.drop(train_dataset.index)

train_features = train_dataset.copy()
test_features = test_dataset.copy()

train_labels = train_features.pop('mpg')
test_labels = test_features.pop('mpg')

data_normalizer = tf.keras.layers.Normalization(axis=1)
data_normalizer.adapt(np.array(train_features))

model = tf.keras.models.Sequential([data_normalizer,Dense(64,activation='relu'),Dense(64,activation='relu'),Dense(1,activation=None)])


model.compile(optimizer=tf.keras.optimizers.Adam(),loss='mean_squared_error')

history = model.fit(x=train_features,y=train_labels,epochs=100,verbose=1,validation_split=0.2)
model.summary()
# plt.plot(history.history['loss'],label='loss')
# plt.plot(history.history['val_loss'],label='val_loss')
# plt.xlabel('Epoch')
# plt.ylabel('Error [MPG]')
# plt.legend()
# plt.grid(True)
# plt.show()

y_pred = model.predict(test_features).flatten()
# a = plt.axes(aspect='equal')
# plt.scatter(test_labels,y_pred)
# plt.xlabel('True Values [MPG]')
# plt.ylabel('Prediction [MPG]')
# lims = [0,50]
# plt.xlim(lims)
# plt.ylim(lims)
# plt.plot(lims,lims)
# plt.show()

error = y_pred - test_labels
plt.hist(error,bins=30)
plt.xlabel('Prediction Error [MPG]')
plt.ylabel('Count')
plt.show()