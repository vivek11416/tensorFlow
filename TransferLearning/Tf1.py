#https://storage.googleapis.com/ztm_tf_course/food_vision/10_food_classes_10_percent.zip

import os

import keras
from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
import tensorflow_hub as hub
from keras import layers
import matplotlib.pyplot as plt


# for dirpath, dirnames, filenames in os.walk("10_food_classes_10_percent"):
#   print(f"There are {len(dirnames)} directories and {len(filenames)} images in '{dirpath}'.")

IMAGE_SHAPE = (224,224)
BATCH_SIZE = 32

train_dir = "10_food_classes_10_percent/train/"
test_dir = "10_food_classes_10_percent/test/"

train_datagen = ImageDataGenerator(rescale=1/255.)
test_datagen = ImageDataGenerator(rescale=1/255.)

print("Training Images : ")
train_data_10_percent = train_datagen.flow_from_directory(train_dir,target_size=IMAGE_SHAPE,batch_size=BATCH_SIZE,class_mode="categorical") # if working with only 2 classes class_mode is binary

print("Testing Images : ")
test_data_10_percent = test_datagen.flow_from_directory(test_dir,target_size=IMAGE_SHAPE,batch_size=BATCH_SIZE,class_mode="categorical")

# callBacks
import datetime

def create_tensorboard_callback(dir_name,experiment_name):
    log_dir = dir_name + "/" + experiment_name + "/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    tensorboad_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir)
    print(f"Saving Tensorboard log files to: {log_dir}")
    return tensorboad_callback

#005 Building and compiling a TensorFlow Hub feature extraction model
resnet_url = "https://kaggle.com/models/google/resnet-v2/frameworks/TensorFlow2/variations/50-feature-vector/versions/1"
efficientnet_url = "https://www.kaggle.com/models/google/efficientnet/frameworks/TensorFlow1/variations/b0-feature-vector/versions/1"

def plot_loss_curves(history):
    loss = history.history["loss"]
    val_loss = history.history["val_loss"]
    accuracy = history.history["accuracy"]
    val_accuracy = history.history["val_accuracy"]

    epochs = range(len(history.history["loss"]))

    plt.plot(epochs,loss,label="training_loss")
    plt.plot(epochs,val_loss,label="val_loss")
    plt.title("loss")
    plt.xlabel("epochs")
    plt.legend()

    #plot  accuracy
    plt.figure()
    plt.plot(epochs,accuracy,label="training_accuracy")
    plt.plot(epochs,val_accuracy,label="val_accuracy")
    plt.title("accuracy")
    plt.xlabel("epochs")
    plt.legend()

def create_model(model_url,num_classes=10):
    feature_extractor_layer = hub.KerasLayer(model_url,trainable=False,name="feature_extractin_layer",input_shape=IMAGE_SHAPE+(3,)) #converts (224,224) to (224,224,3)#trainable false freezes model
    model = keras.Sequential([
        feature_extractor_layer,
    layers.Dense(num_classes,activation="softmax",name="output_layer")])
    return model


#create Resnet Model
resnet_model = create_model(resnet_url,num_classes=train_data_10_percent.num_classes)
resnet_model.summary()

resnet_model.compile(loss="categorical_crossentropy",optimizer=tf.optimizers.Adam(),metrics=['accuracy'])

#006 Blowing our previous models out of the water with transfer learning

history_resnet_model = resnet_model.fit(train_data_10_percent,epochs=5,steps_per_epoch=len(train_data_10_percent),validation_data=test_data_10_percent,
                                        validation_steps=len(test_data_10_percent),
                                        callbacks=[create_tensorboard_callback(dir_name='tensorflow_hub',experiment_name='resnet50v2')])





