#https://storage.googleapis.com/ztm_tf_course/food_vision/10_food_classes_10_percent.zip

import os
from keras.preprocessing.image import ImageDataGenerator


for dirpath, dirnames, filenames in os.walk(r"C:/Users/z004hpvp/Downloads/10_food_classes_10_percent"):
  print(f"There are {len(dirnames)} directories and {len(filenames)} images in '{dirpath}'.")


