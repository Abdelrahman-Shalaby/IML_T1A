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
from sklearn.metrics import mean_squared_error
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

w = np.linalg.inv(A.T*A)*A.T*y
y_pred = A*w
rms = mean_squared_error(y, y_pred)**0.5
print("No Validation RMSE: ", rms)

k = 10 # we want to to a 10-fold cross-validation 

# Regularization parameters we want to test
lambda_vect = np.array([0.1, 1, 10, 100, 1000])

# How I'm doing the KFolds:
# https://towardsdatascience.com/train-test-split-and-cross-validation-in-python-80b61beca4b6

# TODO: Someting I read online is that it is a good idea to repeat the KFold 
#       experiment more than once, each time shuffling the data and using 
#       different folds. This randomization decreases your chances of 
#       picking a bad fold by accident. We should add this to our script. 
#       KFolds has a nice feature where you can set a shuffle flag which 
#       rearranges the data, and you can also set a random seed for this.

# NOTE: 2018-03-10 - To replicate original results, use one repitition, and no 
#                    shuffling in the call to KFold.
reps = 1
rms_mat = np.matrix(np.zeros((reps, len(lambda_vect))))
for r in range(reps): # we're going to do the entire cross validation process "reps" times
    kf = KFold(n_splits=k, shuffle=False) # define the split into 10 folds, use random shuffling
    rms_vect = np.zeros(np.shape(lambda_vect))

    # Now iterate over the regularization parameters
    for i in range(len(lambda_vect)):
        #print("Testing Lambda = ", lambda_vect[i])

        # this will loop 'k' times
        for train_index, test_index in kf.split(A):
            #print("fold")

            # Extract the training and test data
            A_train, A_test = A[train_index,:], A[test_index,:]
            y_train, y_test = y[train_index,:], y[test_index,:]

            # Compute RMS for current batch
            rms_i = compute_test_rms(A_train, y_train, A_test, y_test, lambda_vect[i])

            # Add to running average
            rms_vect[i] += rms_i

    # take average
    rms_vect /= k

    # Save into matrix of results
    rms_mat[r,:] = rms_vect

print(rms_mat)
rms_out = np.mean(rms_mat, axis=0)
rms_var = np.var(rms_mat, axis=0)

print("Averages: ", rms_out)
print("Variances: ", rms_var) 
# interesting that the last one has a really large variance... I think this explains that c-term in
# the weighting function, and why it gets lower for the last number... They must have anticipated
# this... Veeeeery interesting. 

#############################
#   Write Ouput to File     #
#############################
np.savetxt('results.csv', rms_out.T, fmt='%.12f', newline='\n', comments='')




