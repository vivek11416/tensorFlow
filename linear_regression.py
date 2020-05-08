# -*- coding: utf-8 -*-
"""
Created on Tue May  5 23:36:55 2020

@author: vivek
"""


import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

x_data = tf.linspace(0., 10, 10) + tf.random_uniform([10], -1.5,1.5)
y_data = tf.linspace(0., 10, 10) + tf.random_uniform([10], -1.5,1.5 )

x_test = tf.linspace(-1., 10, 11)


# y = mx+b

m = tf.Variable(0.57)
b = tf.Variable(0.45)

error = 0

for x,y in tf.map_fn(zip(x_data,y_data)):
    y_hat = m*x + b
    
    error += (y-y_hat)**2
    
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
train = optimizer.minimize(error)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    
    training_steps = 1
    
    for i in range(training_steps):
        sess.run(train)
    
    final_slope , final_inercept = sess.run([m,b])
    
    y_pred_plot = final_slope *x_test + final_inercept
    
    plt.plot(sess.run(x_test),sess.run(y_pred_plot),'r')
    plt.plot(sess.run(x_data),sess.run(y_data),'*')
        
    
    