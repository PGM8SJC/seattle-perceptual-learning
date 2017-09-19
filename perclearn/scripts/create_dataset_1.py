# -*- coding: utf-8 -*-

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

X_train, X_train_info = create_new_dataset(X_train, [[0,0]])        
np.savez(opj(cwd, 'perclearn/data/experiments/1/training'), X_train)


### We need to iterate over this and the model:
X_test = create_new_dataset(X_test, [[0,28],[28,0],[28,28]])        
np.savez(opj(cwd, 'perclearn/data/experiments/1/testing'),X_test)  
        
