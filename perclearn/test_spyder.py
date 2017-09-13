# -*- coding: utf-8 -*-

import os
from os.path import join as opj
import numpy as np
import matplotlib.pyplot as plt

from perclearn import mnist_reader
from perclearn.utils import (create_2D_noise,
                             scaler_2D,
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
image = create_2D_noise()
image *= (255.0/image.max())
image = image/(image.max()/255.0)


plt.imshow(a)
plt.imshow(b)

x_offset=y_offset=0
b[y_offset:y_offset+a.shape[0], x_offset:x_offset+a.shape[1]] = a

plt.imshow(b)

