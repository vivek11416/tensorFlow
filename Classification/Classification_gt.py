# intro to neural network classification with tensorflow
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
    tf.keras.layers.Dense(1,activation=tf.keras.activations.relu ),


])

model_1.compile(loss=tf.keras.losses.BinaryCrossentropy(),
                optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
                metrics=["accuracy"],)

model_1.fit(X,y,epochs=100)