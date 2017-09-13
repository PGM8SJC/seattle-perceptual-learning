# -*- coding: utf-8 -*-

import os
from os.path import join as opj
import numpy as np
import matplotlib.pyplot as plt

from perclearn import mnist_reader
from perclearn.utils import (create_2D_noise,
                             scale_2D,
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

result = createCircularMask(a, b, 0, 0)


plt.imshow(result)
plt.imshow(b)

x_offset=y_offset=0
b[y_offset:y_offset+a.shape[0], x_offset:x_offset+a.shape[1]] = a

plt.imshow(b)


plt.imshow(createCircularMask(28, 28, center=None, radius=None))

def create_composition(input_image, background_image,
                       x_offset=0, y_offset=0,
                       center=None, radius=None):

    w, h = input_image.shape
    
    # center and radius calculation for input_image
    if center is None:
        center = [int(w/2), int(h/2)]
        
    if radius is None:
        radius = min(center[0], center[1], w-center[0], h-center[1])

    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)

    mask = dist_from_center <= radius
    
    
    circle_image = np.ma.array(input_image,
                               mask = ~mask)

    out_image = np.ma.array(input_image,
                            mask = mask).data
    
    out_mask = np.ma.array(dist_from_center,
                           mask = mask)
    out_mask_full = out_mask.copy()
    out_mask_full[out_mask<=radius] = radius
    
    dist_from_center_rescaled = scale_2D(out_mask_full.data,
                                         scale_range=(0, 1))

    background_offset = background_image[y_offset:y_offset+w, x_offset:x_offset+h]

    background_fade = background_offset * dist_from_center_rescaled + out_image
    new_input_image = background_fade + circle_image
    
    background_image[y_offset:y_offset+w, x_offset:x_offset+h] = new_input_image

    return background_image
