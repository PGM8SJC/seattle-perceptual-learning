# -*- coding: utf-8 -*-

import os
from os.path import join as opj
import numpy as np
import matplotlib.pyplot as plt

from perclearn import mnist_reader
from perclearn.utils import (create_2D_noise,
                             scale_2D,
                             create_new_dataset,
                             )



cwd = os.getcwd()

X_train, y_train = mnist_reader.load_mnist(opj(cwd, 'perclearn/data/fashion'),
                                           kind='train')
X_test, y_test = mnist_reader.load_mnist(opj(cwd, 'perclearn/data/fashion'),
                                         kind='t10k')

X_train = create_new_dataset(X_train, [[0,0]])        
np.savez(opj(cwd, 'perclearn/data/experiments/1/training'),X_train)

X_test = create_new_dataset(X_test, [[0,28],[28,0],[28,28]])        
np.savez(opj(cwd, 'perclearn/data/experiments/1/testing'),X_test)
    


dataset = X_test
i = 10
offsets=[[0,28],[28,0],[28,28]]

m, n = dataset.shape
im_x = int(np.sqrt(n))
num_offsets = len(offsets)

new_dataset = np.zeros((m,n*4))

image = np.reshape(dataset[i,:], (im_x,im_x))
noise_bg = scale_2D(create_2D_noise())

rand_offset = np.random.randint(0,num_offsets)

result = create_composition(image, noise_bg,
               x_offset=offsets[rand_offset][0],
               y_offset=offsets[rand_offset][1],
               center=None, radius=None)
plt.imshow(result)   

"""
Testing week 2
"""
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
 
#let's generate some examples of radial noise
dataset = X_test
i = 10
offsets=[[0,0]]
m, n = dataset.shape
im_x = int(np.sqrt(n))
new_dataset = np.zeros((m,n*4))
image = np.reshape(dataset[i,:], (im_x,im_x))
noise_bg = scale_2D(create_2D_noise())

for i in range(0,15,2):
    result = create_composition(image, noise_bg,
                   x_offset=0,
                   y_offset=0,
                   center=None, radius=i)
    plt.figure()
    plt.imshow(result)


result = create_composition(image, noise_bg,
                           x_offset=0,
                           y_offset=0,
                           center=None, radius=None)
plt.figure()
plt.imshow(result)  

"""
Create new dataset
"""
import os
from os.path import join as opj
import numpy as np

from perclearn import mnist_reader
from perclearn.utils import (create_new_dataset)

cwd = os.getcwd()

X_train, y_train = mnist_reader.load_mnist(opj(cwd, 'perclearn/data/fashion'),
                                           kind='train')
X_test, y_test = mnist_reader.load_mnist(opj(cwd, 'perclearn/data/fashion'),
                                         kind='t10k')

X_train, X_train_info = create_new_dataset(X_train[:10,:], [[0,0]], rotate_bool=True)        
np.savez(opj(cwd, 'perclearn/data/experiments/1/training'), X_train)

X_test = create_new_dataset(X_test, [[0,28],[28,0],[28,28]])        
np.savez(opj(cwd, 'perclearn/data/experiments/1/testing'),X_test)  
        
"""
"""
import os
from os.path import join as opj
import numpy as np
import matplotlib.pyplot as plt

from perclearn import mnist_reader
from perclearn.utils import (create_new_dataset)

cwd = os.getcwd()

# the data, shuffled and split between train and test sets
x_train_full, y_train_full = mnist_reader.load_mnist(opj(cwd, 'perclearn/data/fashion'),
                                           kind='train')


offsets = [[x,y] for x in range(28) for y in range(28)]

# Generating of x_train and dividing in x_train and x_val
x, _ = create_new_dataset(x_train_full[0:2,:], offsets=offsets,
                                         rotate_bool=True)  
plt.imshow(np.reshape(x[0,:], (56,56)))


for i in range(10):
    plt.figure()
    plt.imshow(np.reshape(x[i,:], (56,56)))



