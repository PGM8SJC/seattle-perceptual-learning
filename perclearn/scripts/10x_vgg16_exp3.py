# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
'''
Trains a simple convnet on the Fashion MNIST dataset modified:
    
    - Experiment 3: Training with the random image generation.
    Testing moving test sets 1 pixel to the right each time
    or rotation 10º.
    
    10x with VGG16

Gets to % test accuracy after 12 epochs
'''

from __future__ import print_function
import keras
from keras.models import Model
from keras.layers import (Input, Dense, Dropout, Flatten, 
                          Conv2D, MaxPooling2D)
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
x_train_full, y_train_full = mnist_reader.load_mnist(opj(cwd, 'perclearn/data/fashion'),
                                           kind='train')
x_test_full, y_test = mnist_reader.load_mnist(opj(cwd, 'perclearn/data/fashion'),
                                         kind='t10k')

offsets = [[x,y] for x in range(28) for y in range(28)]

score_translate_10x = np.empty((10,2,28))
score_rotate_10x = np.empty((10,2,36))

for exp_time in range(10):
    # Generating of x_train and dividing in x_train and x_val
    x_train_full, _ = create_new_dataset(x_train_full, offsets=offsets,
                                             rotate_bool=True)      
    
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
    
    input_tensor = Input(input_shape)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(input_tensor)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

    # Block 2
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

    # Block 3
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

    # Block 4
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

    # Classification block
    x = Flatten(name='flatten')(x)
    x = Dense(4096, activation='relu', name='fc1')(x)
    x = Dense(4096, activation='relu', name='fc2')(x)
    x = Dropout(0.5)(x)
    x = Dense(10, activation='softmax', name='predictions')(x)
    model = Model(inputs=[input_tensor],outputs=[x])
    
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['acc'])
    
    model.fit(x_train, y_train,
              validation_data=(x_val, y_val),
              batch_size=batch_size,
              epochs=epochs,
              verbose=1)    

    
    """
    TRANSLATION
    """
    
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
        
        score_translate_10x[exp_time,:,offset_x] = np.array(model.evaluate(x_test, y_test, verbose=0))
            
    
    """
    ROTATION
    """

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
        
        score_rotate_10x[exp_time,:,offset_x] = np.array(model.evaluate(x_test, y_test, verbose=0))
        
np.savez(opj(cwd, 'perclearn/data/results/translate_10x_vgg16'), score_translate_10x)  
np.savez(opj(cwd, 'perclearn/data/results/rotate_10x_vgg16'), score_rotate_10x) 


