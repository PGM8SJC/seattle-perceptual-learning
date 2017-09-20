# -*- coding: utf-8 -*-
import os
from os.path import join as opj
import numpy as np
import matplotlib.pyplot as plt

cwd = os.getcwd()

# results[0] = test loss; results[1] = test accuracy
translate_results = np.load(opj(cwd, 'perclearn/data/results/translate.npz'))['arr_0']
rotate_results = np.load(opj(cwd, 'perclearn/data/results/rotate.npz'))['arr_0']

# Translation plots
x = np.linspace(0, 28, 28)

f, axarr = plt.subplots(2, sharex=True)
axarr[0].scatter(x, translate_results[0,:])
axarr[0].set_title('Test Loss - Translation')
axarr[1].scatter(x, translate_results[1,:])
axarr[1].set_title('Test Accuracy - Translation')
plt.savefig('/home/asier/Desktop/translate.png')
# Rotation plots
plt.figure()
x = np.linspace(0, 360, 36)

f, axarr = plt.subplots(2, sharex=True)
axarr[0].scatter(x, rotate_results[0,:])
axarr[0].set_title('Test Loss - Rotation')
axarr[1].scatter(x, rotate_results[1,:])
axarr[1].set_title('Test Accuracy - Rotation')
plt.savefig('/home/asier/Desktop/rotate.png')
