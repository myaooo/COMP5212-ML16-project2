# 
# utils.py
# project2
# ming yao
#

import numpy as np
import math

def normalize_image(image, adjust):
    stddev = np.std(image)
    adjust_stddev = max(stddev, adjust)
    mean = np.mean(image)
    return (image -mean)/adjust_stddev

def normalize_data(data):
    # Assume input data matrix X of size [N x D]
    X = data
    X -= np.mean(X, axis = 0) # zero-center the data (important)
    X /= np.std(X, axis = 0)
    return X

def whitening(data):
    # Assume input data matrix data of size [N x D]
    X = data
    X -= np.mean(X, axis = 0) # zero-center the data (important)
    cov = np.dot(X.T, X) / X.shape[0] # get the data covariance matrix

    U,S,_ = np.linalg.svd(cov)
    Xrot = np.dot(X, U) # decorrelate the data
    # whiten the data:
    # divide by the eigenvalues (which are square roots of the singular values)
    Xwhite = Xrot / np.sqrt(S + 1e-6)

    # TestWhite = test_data - np.mean(test_data, axis=0)
    # TestWhite = np.dot(TestWhite, U)
    # TestWhite = TestWhite / np.sqrt(S+1e-5)
    
    return Xwhite

