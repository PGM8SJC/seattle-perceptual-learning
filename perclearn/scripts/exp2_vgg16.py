# -*- coding: utf-8 -*-
'''
Trains a simple convnet on the Fashion MNIST dataset modified:
    
    - Experiment 2: Training with the image in upper left cuadrant 
    and testing moving test sets 1 pixel to the right each time
    or rotation 10º.

Gets to % test accuracy after 12 epochs
'''

from __future__ import print_function
import keras
from keras.applications import VGG16
from keras import backend as K
import numpy as np
import os
from os.path import join as opj

from perclearn import mnist_reader
from perclearn.utils import create_new_dataset

cwd = os.getcwd()

batch_size = 128
num_classes = 10
epochs = 12

# input image dimensions
img_rows, img_cols = 56, 56

# the data, shuffled and split between train and test sets
_, y_train_full = mnist_reader.load_mnist(opj(cwd, 'perclearn/data/fashion'),
                                           kind='train')
x_test_full , y_test = mnist_reader.load_mnist(opj(cwd, 'perclearn/data/fashion'),
                                         kind='t10k')

# Loading of x_train and dividing in x_train and x_val
x_train = np.load(opj(cwd, 'perclearn/data/experiments/1/training.npz'))
x_train_full = x_train['arr_0']

x_train = x_train_full[:50000]
x_val = x_train_full[50000:]

y_train = y_train_full[:50000]
y_val = y_train_full[50000:]


if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_val = x_val.reshape(x_val.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_val = x_val.reshape(x_val.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

x_train = x_train.astype('float32')
x_train /= 255
x_val = x_val.astype('float32')
x_val /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_val = keras.utils.to_categorical(y_val, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

## VGG16 Model
model = VGG16(weights=None,
              input_shape=input_shape,
              pooling=max,
              classes=num_classes)

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adam(),
              metrics=['accuracy'])

model.fit(x_train, y_train,
          validation_data=(x_val, y_val),
          batch_size=batch_size,
          epochs=epochs,
          verbose=1) 


"""
TRANSLATION
"""

score = np.empty((2,28))
# iterate over creation of different x_tests and evaluate model
for offset_x in range(28):
            
    x_test, _ = create_new_dataset(x_test_full, [[offset_x,0]])        
    
    if K.image_data_format() == 'channels_first':
        x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    else:
        x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    
    x_test = x_test.astype('float32')
    x_test /= 255
    print(x_test.shape[0], 'test samples')
    print('offset x: ', offset_x)    
    
    score[:,offset_x] = np.array(model.evaluate(x_test, y_test, verbose=0))
    print('Test loss:', score[0, offset_x])
    print('Test accuracy:', score[1, offset_x])
        
np.savez(opj(cwd, 'perclearn/data/results/translate'), score)  

"""
ROTATION
"""
score = np.empty((2,36))
# iterate over creation of different x_tests and evaluate model
for i, angle in enumerate(range(0,360,10)):
            
    x_test, _ = create_new_dataset(x_test_full, [[0,0]],
                                   rotate_bool=True, angle=angle)      
    
    if K.image_data_format() == 'channels_first':
        x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    else:
        x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    
    x_test = x_test.astype('float32')
    x_test /= 255
    print(x_test.shape[0], 'test samples')
    print('angle: ', angle)    
    
    score[:,i] = np.array(model.evaluate(x_test, y_test, verbose=0))
    print('Test loss:', score[0, i])
    print('Test accuracy:', score[1, i])
        
np.savez(opj(cwd, 'perclearn/data/results/rotate'), score) 