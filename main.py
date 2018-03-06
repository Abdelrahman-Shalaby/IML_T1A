#################################
#          IML_T1A              #
#################################
#
# File Name: main.py
# Course: 252-0220-00L Introduction to Machine Learning
#
# Authors: Adrian Esser (aesser@student.ethz.ch)
#          Abdelrahman-Shalaby (shalabya@student.ethz.ch)

import numpy as np
import sys
import os
import csv
from sklearn.model_selection import KFold 
from helpers import * # import the file with helper functions


# Import training data
data = np.genfromtxt('train.csv', delimiter=',')
data = np.delete(data, 0, 0) # remove first row
data = np.matrix(data)

# Extract data into matrices
A = data[:,2:] # get third column to end
y = data[:,1] # get second column
d = np.shape(A)[1] # number of parameters (should be 10)
n = np.shape(A)[0] # number of data points (should be 500)

k = 10 # we want to to a 10-fold cross-validation 

# Regularization parameters we want to test
lambda_vect = np.array([0.1, 1, 10, 100, 1000])
rms_vect = np.zeros(np.shape(lambda_vect))

# How I'm doing the KFolds:
# https://towardsdatascience.com/train-test-split-and-cross-validation-in-python-80b61beca4b6
kf = KFold(n_splits=k) # define the split into 2 folds
kf.get_n_splits(A) # return the number of splitting iterations in the cross validator

for i in range(len(lambda_vect)):
    # this will loop 'k' times
    for train_index, test_index in kf.split(A):
        # Extract the training and test data
        A_train, A_test = A[train_index,:], A[test_index,:]
        y_train, y_test = y[train_index,:], y[test_index,:]

        # TODO: Shalaby you need to write this function. It's documented in the helper file
        rms_i = compute_test_rms(A_train, y_train, A_test, y_test, lambda_vect[i])
        #rms_i = 1

        rms_vect[i] += rms_i

# take average
rms_vect /= k

#############################
#   Write Ouput to File     #
#############################
np.savetxt('results.csv', rms_vect, fmt='%.12f', newline='\n', comments='')




