#working with larger example (multiclass classification)
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
(train_data,train_labels),(test_data,test_labels) =  tf.keras.datasets.fashion_mnist.load_data()

#check Shape
print(train_data[0].shape , train_labels[0].shape)

class_names = ["T-shirt/top","Trouser","Pullover","Dress","Coat","Sandal","Shirt","Sneaker","Bag","Ankle boot"]

# building a multiclass classification model
#tf.random.set_seed(42)

model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28,28)),
    tf.keras.layers.Dense(4,activation=tf.keras.activations.relu),
    tf.keras.layers.Dense(4,activation=tf.keras.activations.relu),
    tf.keras.layers.Dense(10,activation=tf.keras.activations.softmax),
])

model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(), #if your labels are one hot encoded, use categoricalcrossentropy, if integer use sparseCategoricalEntropy
              optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              metrics=["accuracy"])

non_norm_history= model.fit(train_data,train_labels,epochs=10,validation_data=(test_data,test_labels))

train_data_norm = train_data / train_data.max()
test_data_norm = test_data / test_data.max()

tf.random.set_seed(42)

model_norm = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28,28)),
    tf.keras.layers.Dense(4,activation=tf.keras.activations.relu),
    tf.keras.layers.Dense(2,activation=tf.keras.activations.relu),
    tf.keras.layers.Dense(10,activation=tf.keras.activations.softmax),
])

model_norm.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(), #if your labels are one hot encoded, use categoricalcrossentropy, if integer use sparseCategoricalEntropy
              optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              metrics=["accuracy"])

norm_history= model.fit(train_data_norm,train_labels,epochs=10,validation_data=(test_data_norm,test_labels))


pd.DataFrame(non_norm_history.history).plot(title="Non-normalized data")
pd.DataFrame(norm_history.history).plot(title="Normalized data")
plt.show()




