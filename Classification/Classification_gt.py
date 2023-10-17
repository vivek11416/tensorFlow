# intro to neural network classification with tensorflow
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import make_circles
import tensorflow as tf
#Make 1000 examples
n_samples = 1000

#Create circles
X,y=make_circles(n_samples,
                 noise=0.03,
                 random_state=42,)

tf.random.set_seed(42)

model_1 = tf.keras.Sequential([
    tf.keras.layers.Dense(4,activation=tf.keras.activations.relu ),
tf.keras.layers.Dense(4,activation=tf.keras.activations.relu ),
tf.keras.layers.Dense(1,activation=tf.keras.activations.sigmoid ),


])

model_1.compile(loss=tf.keras.losses.BinaryCrossentropy(),
                optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
                metrics=["accuracy"],)

callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)

#model_1.fit(X,y,epochs=50,callbacks=callback)

tf.random.set_seed(43)

model_2 = tf.keras.Sequential([
    tf.keras.layers.Dense(4,activation=tf.keras.activations.relu ),
    tf.keras.layers.Dense(4,activation=tf.keras.activations.relu ),
    tf.keras.layers.Dense(1,activation=tf.keras.activations.sigmoid ),

])

model_2.compile(loss=tf.keras.losses.BinaryCrossentropy(),
                optimizer=tf.keras.optimizers.Adam(),
                metrics=["accuracy"],)

#Callback runs/works during model training

lr_scheduler =  tf.keras.callbacks.LearningRateScheduler(lambda epoch: 1e-4 * 10**(epoch/20))

hist_2= model_2.fit(X,y,epochs=100,callbacks=[lr_scheduler])

#check history
pd.DataFrame(hist_2.history).plot(figsize=(10,7),xlabel="epochs")
plt.show()

