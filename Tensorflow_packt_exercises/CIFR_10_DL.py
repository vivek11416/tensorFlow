import numpy as np
import tensorflow as tf
from keras import datasets,layers,models,optimizers

IMG_CHANNELS = 3
IMG_ROWS = 32
IMG_COLS = 32

INPUT_SHAPE = (IMG_ROWS,IMG_COLS,3)

BATCH_SIZE = 128
EPOCHS = 50
CLASSES = 10
VERBOSE = 1
VALIDATION_SPLIT = 0.2
OPTIM = tf.keras.optimizers.RMSprop()

def build(input_shape,classes):
    model = tf.keras.models.Sequential()
    model.add(layers.Conv2D(32,(3,3),activation='relu',input_shape=input_shape))
    model.add(layers.MaxPooling2D(pool_size=(2,2)))
    model.add(layers.Dropout(0.25))
    model.add(layers.Flatten())
    model.add(layers.Dense(512,activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(classes,activation='softmax'))
    return model

def build_model(X_train,CLASSES):
    model = models.Sequential()

    model.add(layers.Conv2D(32,(3,3),padding='same',input_shape=X_train.shape[1:],activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(32,(3,3),padding='same',activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D(pool_size=(2,2)))
    model.add(layers.Dropout(0.2))

    model.add(layers.Conv2D(64,(3,3),padding='same',activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(64,(3,3),padding='same',activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D(pool_size=(2,2)))
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(128, (3, 3), padding='same', activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(128, (3, 3), padding='same', activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Dropout(0.4))
    model.add(layers.Flatten())
    model.add(layers.Dense(CLASSES,activation='softmax'))
    return model

def load_data():
    (X_train,y_train),(X_test,y_test) = tf.keras.datasets.cifar10.load_data()
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')

    mean = np.mean(X_train,axis=(0,1,2,3))
    std = np.std(X_train,axis=(0,1,2,3))
    X_train = (X_train-mean)/(std+1e-7)
    X_test = (X_test-mean)/(std+1e-7)

    y_train=tf.keras.utils.to_categorical(y_train,CLASSES)
    y_test = tf.keras.utils.to_categorical(y_test, CLASSES)

    return X_train,y_train,X_test,y_test

callbacks = [tf.keras.callbacks.TensorBoard(log_dir='./logs')]

#for build_model method
(X_train,y_train,X_test,y_test) = load_data()
model=build_model(X_train,CLASSES)
model.compile(loss='categorical_crossentropy',optimizer='RMSprop',metrics=['accuracy'])
batch_size = 64
model.fit(X_train,y_train,batch_size=batch_size,epochs=EPOCHS,validation_data=(X_test,y_test),callbacks=callbacks)
score = model.evaluate(X_test,y_test,batch_size=batch_size)
print("\nTest score:",score[0])
print('Test accuracy:',score[1])






#For normal build method
# (X_train,y_train),(X_test,y_test) = tf.keras.datasets.cifar10.load_data()
#
# X_train = tf.cast(X_train,tf.float32)
# X_test = tf.cast(X_test,tf.float32)
#
# X_train , X_test = X_train/255.0,X_test/255.0
#
# y_train = tf.keras.utils.to_categorical(y_train,CLASSES)
# y_test = tf.keras.utils.to_categorical(y_test,CLASSES)
#
# model = build(INPUT_SHAPE,CLASSES)
# model.compile(loss="categorical_crossentropy",optimizer=OPTIM,metrics=["accuracy"])
# model.summary()
#
# model.fit(X_train,y_train,batch_size=BATCH_SIZE,epochs=EPOCHS,validation_split=VALIDATION_SPLIT,verbose=VERBOSE,callbacks=callbacks)
# score = model.evaluate(X_test,y_test,batch_size=BATCH_SIZE,verbose=VERBOSE)
#
# print("\nTest score:",score[0])
# print('Test accuracy:',score[1])

