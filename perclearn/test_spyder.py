# -*- coding: utf-8 -*-

import os
from os.path import join as opj
import numpy as np
import matplotlib.pyplot as plt

from perclearn import mnist_reader
from perclearn.utils import (create_2D_noise,
                             scale_2D,
                             create_composition,
                             )



cwd = os.getcwd()

X_train, y_train = mnist_reader.load_mnist(opj(cwd, 'perclearn/data/fashion'),
                                           kind='train')
X_test, y_test = mnist_reader.load_mnist(opj(cwd, 'perclearn/data/fashion'),
                                         kind='t10k')

print(X_train.shape)

plt.imshow(np.reshape(X_train[0,:], (28,28)), cmap=plt.cm.Greys)
print(y_train[0])

a = np.reshape(X_train[0,:], (28,28))
b = X_train[0,:]

# Flattening function
np.ndarray.flatten()

plt.imshow(create_2D_noise())


a = np.reshape(X_train[0,:], (28,28))
b = scale_2D(create_2D_noise())

result = create_composition(a, b,
                       x_offset=0, y_offset=0,
                       center=None, radius=None)


plt.imshow(result)

def create_new_dataset(dataset, offsets=[[0,0]]):
    
    m, n = dataset.shape
    im_x = int(np.sqrt(n))
    num_offsets = len(offsets)
    
    new_dataset = np.zeros((m,n*4))
    
    for i in range(m):
        image = np.reshape(X_train[i,:], (im_x,im_x))
        noise_bg = scale_2D(create_2D_noise())
        
        rand_offset = np.random.randint(0,num_offsets)

        result = create_composition(image, noise_bg,
                       x_offset=offsets[rand_offset,0],
                       y_offset=offsets[rand_offset,1],
                       center=None, radius=None)
        
        new_dataset[i,:] = np.ndarray.flatten(result)
        
    return new_dataset
    
    
    
    
    
    
    
    
    
    