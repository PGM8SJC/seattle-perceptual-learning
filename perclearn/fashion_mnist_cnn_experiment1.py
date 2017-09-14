# -*- coding: utf-8 -*-
'''Trains a simple convnet on the Fashion MNIST dataset modified:
    
    - Experiment 1: Training with the image in upper left cuadrant 
    and testing randomly distributed among other cuadrants.

Gets to % test accuracy after 12 epochs
'''

from __future__ import print_function
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
import numpy as np
import os
from os.path import join as opj

from perclearn import mnist_reader


cwd = os.getcwd()

batch_size = 128
num_classes = 10
epochs = 12

# input image dimensions
img_rows, img_cols = 56, 56

# the data, shuffled and split between train and test sets
X_train, y_train = mnist_reader.load_mnist(opj(cwd, 'perclearn/data/fashion'),
                                           kind='train')
X_test, y_test = mnist_reader.load_mnist(opj(cwd, 'perclearn/data/fashion'),
                                         kind='t10k')

x_train = np.load(opj(cwd, 'perclearn/data/experiments/1/training.npz'))
x_train = x_train['arr_0']

x_test = np.load(opj(cwd, 'perclearn/data/experiments/1/testing.npz'))
x_test = x_test['arr_0']


if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test))
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

