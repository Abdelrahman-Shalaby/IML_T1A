import numpy as np
import sys
import os
from sklearn.metrics import mean_squared_error
from math import sqrt

def compute_test_rms(A_train, y_train, A_test, y_test, lam):
    # Compute the LS weights using the training data 
    w_ridge = np.linalg.inv(A_train.T*A_train + lam*np.identity(d)) * A.T*y_train
    
    #Use the weights to evaluate the predictions on the test set
    y_predicted = np.dot(A_test, w_ridge)
    
    # Using y_test and the predictions, compute the RMS error
    rms = sqrt(mean_squared_error(y_test, y_predicted))
    
    # Return this RMS value
    return rms

