import sklearn
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_moons
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy
X,y =  make_moons(n_samples=5000, shuffle=True, noise=None, random_state=None)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)

def plot_data(X,y):

    # When the label y is 0, the class is represented with a blue square.
    # When the label y is 1, the class is represented with a green triangle.
    plt.plot(X[:, 0][y==1], X[:, 1][y==1], "bs")
    plt.plot(X[:, 0][y==0], X[:, 1][y==0], "g^")

    # X contains two features, x1 and x2
    plt.xlabel(r"$x_1$", fontsize=20)
    plt.ylabel(r"$x_2$", fontsize=20)

    # Simplifying the plot by removing the axis scales.
    plt.xticks([])
    plt.yticks([])

    # Displaying the plot.
    plt.show()

model = tf.keras.Sequential([
    tf.keras.layers.Dense(4,activation=tf.keras.activations.relu),
    tf.keras.layers.Dense(4,activation=tf.keras.activations.relu),
    tf.keras.layers.Dense(1,activation=tf.keras.activations.sigmoid)
])

model.compile(loss=tf.keras.losses.BinaryCrossentropy(),
              optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              metrics=['accuracy'])

model.fit(X_train,y_train,epochs=30)

y_pred = model.predict(X_test)
y_pred_labels = numpy.array([1 if x >=0.5 else 0 for x in y_pred])
# print(type(numpy.array(y_pred_labels)))
# print(type(y_test))

print(sklearn.metrics.confusion_matrix(y_test,y_pred_labels))

def softmax(arr):
    sumExp = sum([tf.math.exp(x) for x in arr])
    return tf.constant([(tf.math.exp(y)/sumExp).numpy() for y in arr])

print(softmax(arr=[2.0,5.0,7.0,8.0,9.,3.,2.,4.,54.,6.57]))