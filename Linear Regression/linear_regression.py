# -*- coding: utf-8 -*-
"""
Created on Tue May  5 23:36:55 2020

@author: vivek
"""
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf



# Creating data to view and fit
X = np.array([-7.0, -4.0, -1.0, 2.0, 5.0, 8.0, 11.0, 14.0])

y = np.array([3.0,6.0,9.0,12.0,15.0,18.0,21.0,24.0])

#Visualize
plt.scatter(X,y)
#plt.show()

house_info = tf.constant(["bedroom","bathroom","garage"])
house_price = tf.constant([939700])
house_info,house_price

X = tf.constant(X)
y= tf.constant(y)

tf.random.set_seed(42)

model = tf.keras.Sequential([
    tf.keras.layers.Dense(50,activation=None,trainable=False),
    tf.keras.layers.Dense(1)
])

model.compile(loss=tf.keras.losses.mae,
              optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
              metrics=['mae'])

model.fit(tf.expand_dims(X,axis=-1),y,epochs=100)

print(model.summary())
# tf.keras.utils.plot_model(model)

tf.keras.losses.MAPE()






