import itertools
import os
import random

import cv2
import numpy as np
import pandas as pd
import sklearn.metrics as metrics
import tensorflow as tf
from kerastuner.tuners import Hyperband
from sklearn.metrics import (classification_report, confusion_matrix,
                             plot_confusion_matrix)
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras import layers, optimizers, regularizers
# Need to edit below import if using base model other than DenseNet
# Need to edit below import if using base model other than DenseNet
from tensorflow.keras.applications.densenet import (DenseNet201,
                                                    decode_predictions,
                                                    preprocess_input)
from tensorflow.keras.callbacks import CSVLogger, ModelCheckpoint
from tensorflow.keras.layers import (Activation, BatchNormalization, Conv2D,
                                     Dense, Dropout, Flatten,
                                     GlobalAveragePooling2D, MaxPooling2D)
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img

device_name = tf.test.gpu_device_name()
if device_name != '/device:GPU:0':
   raise SystemError('GPU device not found')
print('Found GPU at: {}'.format(device_name))


class Sample:
   def _init_(self, directory_path):
      self.directory_path = directory_path

   def get_image_paths(self):
      list_image_paths = [os.path.join(self.directory_path, label, image) for label in os.listdir(
          self.directory_path) for image in os.listdir(os.path.join(self.directory_path, label))]
      return list_image_paths

   def get_target_labels(self):
      list_target_labels = [label for label in os.listdir(
          self.directory_path) for image in os.listdir(os.path.join(self.directory_path, label))]
      return list_target_labels


def augment(image, label):
   image = tf.image.random_flip_left_right(image)
   image = tf.image.random_brightness(image, max_delta=0.5)
   #image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
   image = tf.image.random_crop(image, size=[224, 224, 3])
   rotation = random.randint(0, 3)
   image = tf.image.rot90(image, rotation)

   return image, label


def load_and_augment(image_path, label):
   image = tf.io.read_file(image_path)
   image = tf.image.decode_jpeg(image, channels=3)
   image = tf.image.resize(image, (224, 224))
   image /= 255
   return augment(image, label)


def only_load(image_path):
   image = tf.io.read_file(image_path)
   image = tf.image.decode_jpeg(image, channels=3)
   image = tf.image.resize(image, (224, 224))
   image /= 255
   return image


def build_model(hp):
   base_model = DenseNet201(include_top=False, weights='imagenet')

   x = base_model.output
   x = GlobalAveragePooling2D()(x)

   x = Dense(hp.Choice('hidden_units1', values=[
             32, 64, 128, 256]), activation='relu')(x)
   x = Dropout(hp.Choice('dropout_1', values=[0.1, 0.2, 0.3, 0.4, 0.5]))(x)
   x = Dense(hp.Choice('hidden_units2', values=[
             32, 64, 128, 256]), activation='relu')(x)
   x = Dropout(hp.Choice('dropout_2', values=[0.1, 0.2, 0.3, 0.4, 0.5]))(x)

   predictions = Dense(9, activation='softmax')(x)
   model = Model(inputs=base_model.input, outputs=predictions)

   model.compile(optimizer=SGD(lr=hp.Choice('learning_rate', values=[0.0001, 0.001, 0.01, 0.1]), momentum=0.9),
                 loss='categorical_crossentropy', metrics=['accuracy'])

   return model


np.random.seed(61)
random.seed(61)
tf.random.set_seed(61)

with tf.device('/device:GPU:1'):
   Sample_object = Sample(r"/home/devthumos/FSI/Dataset")
   list_image_paths = Sample_object.get_image_paths()
   list_target_labels = Sample_object.get_target_labels()
   list_target_dummies = pd.get_dummies(list_target_labels)
   X_train, X_test, Y_train, Y_test = train_test_split(
       list_image_paths, list_target_dummies, test_size=0.2, random_state=42)

   # Split the training data into training and validation datasets
   X_train, X_val, Y_train, Y_val = train_test_split(
       X_train, Y_train, test_size=0.2, random_state=42)
   # Test set
   img_ds_test = tf.data.Dataset.from_tensor_slices(X_test)
   img_ds_test = img_ds_test.map(
       only_load, num_parallel_calls=tf.data.AUTOTUNE)
   img_array_test = np.array([img.numpy() for img in img_ds_test])
   X_test = img_array_test
   Y_test = np.array(Y_test)

   # Validation set
   img_ds_val = tf.data.Dataset.from_tensor_slices(X_val)
   img_ds_val = img_ds_val.map(only_load, num_parallel_calls=tf.data.AUTOTUNE)
   img_array_val = np.array([img.numpy() for img in img_ds_val])
   X_val = img_array_val
   Y_val = np.array(Y_val)

   # Combine the image paths and labels into a dataset
   image_label_ds_train = tf.data.Dataset.from_tensor_slices(
       (X_train, Y_train))

   # Apply the load and augmentation function to each image in the dataset
   augmented_image_label_ds_train = image_label_ds_train.map(
       load_and_augment, num_parallel_calls=tf.data.AUTOTUNE)

   # Repeat the dataset multiple times to increase its size
   augmented_image_label_ds_train = augmented_image_label_ds_train.repeat(1)

   # Shuffle the dataset
   augmented_image_label_ds_train = augmented_image_label_ds_train.shuffle(
       buffer_size=100)
   # Convert the dataset to a numpy array
   augmented_image_label_array_train = pd.DataFrame(
       [(img.numpy(), label.numpy()) for img, label in augmented_image_label_ds_train])
   X_train = augmented_image_label_array_train[0]
   Y_train = pd.DataFrame(augmented_image_label_array_train[1].tolist(), columns=[
                          i for i in range(9)])
   Y_train = np.array(Y_train)
   X_train = np.array([np.reshape(sample, (224, 224, 3))
                      for sample in X_train])
   tuner = Hyperband(build_model, objective='val_accuracy',
                     max_epochs=30, factor=3)
   stop_early = keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
   tuner.search(X_train, Y_train, epochs=1, validation_data=(
       X_val, Y_val), callbacks=[stop_early])
   best_hyperparameters = tuner.get_best_hyperparameters()[0]
   best_model = tuner.get_best_models()[0]
   best_model.save("/home/devthumos/FSI/Weigths/best_model_luis.h5")
