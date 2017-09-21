# -*- coding: utf-8 -*-
import os
from os.path import join as opj
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

cwd = os.getcwd()

# results[0] = test loss; results[1] = test accuracy
# 2CV
translate_10x_2cv_rand = np.load(opj(cwd, 'perclearn/data/results/translate_10x_2cv.npz'))['arr_0']
rotate_10x_2cv_rand = np.load(opj(cwd, 'perclearn/data/results/rotate_10x_2cv.npz'))['arr_0']

translate_2cv_fixed = np.load(opj(cwd, 'perclearn/data/results/translate.npz'))['arr_0']
rotate_2cv_fixed= np.load(opj(cwd, 'perclearn/data/results/rotate.npz'))['arr_0']

# VGG16
translate_10x_vgg16_rand = np.load(opj(cwd, 'perclearn/data/results/translate_10x_vgg16.npz'))['arr_0']
rotate_10x_vgg16_rand = np.load(opj(cwd, 'perclearn/data/results/rotate_10x_vgg16.npz'))['arr_0']

translate_vgg16_fixed = np.load(opj(cwd, 'perclearn/data/results/translate_exp2_vgg16.npz'))['arr_0']
rotate_vgg16_fixed= np.load(opj(cwd, 'perclearn/data/results/rotate_exp2_vgg16.npz'))['arr_0']


"""
plots
"""
# TRANSLATE
sns.tsplot(translate_10x_2cv_rand[:,1,:])
plt.plot(translate_2cv_fixed[1,:])
sns.tsplot(translate_10x_vgg16_rand[:,1,:])
plt.plot(translate_vgg16_fixed[1,:])
plt.legend(['Fixed 2cv', 'Random 10x_2cv',
            'Fixed vgg16', 'Random 10x_vgg16'])
plt.title('TRANSLATION')
plt.xlabel('Offset x axis')
plt.ylabel('Accuracy')

plt.savefig('/home/asier/Desktop/translate_2cv_vs_vgg16.png')
# Rotation plots



#ROTATE
plt.figure()
x = np.linspace(0, 360, 36)

sns.tsplot(rotate_10x_2cv_rand[:,1,:], time=x)
plt.plot(x, rotate_2cv_fixed[1,:])
sns.tsplot(translate_10x_vgg16_rand[:,1,:], time=x)
plt.plot(translate_vgg16_fixed[1,:])
plt.legend(['Fixed 2cv', 'Random 10x_2cv',
            'Fixed vgg16', 'Random 10x_vgg16'])
plt.title('ROTATION')
plt.xlabel('Angle rot')
plt.ylabel('Accuracy')

plt.savefig('/home/asier/Desktop/rotate_2cv_vs_vgg16.png')

