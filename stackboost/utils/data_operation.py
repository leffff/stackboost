from __future__ import division
import numpy as np
import math
from numba import jit

# @jit(nopython=True)
def calculate_entropy(y):
    """ Calculate the entropy of label array y """
    log2 = lambda x: math.log(x) / math.log(2)
    unique_labels = np.unique(y)
    entropy = 0
    for label in unique_labels:
        count = len(y[y == label])
        p = count / len(y)
        entropy += -p * log2(p)
    return entropy


@jit(nopython=True)
def similarity_score(y, reg):
    return (np.sum(y) ** 2) / (len(y) + reg)


def similarity_gain(y_root, y_left, y_right, reg):
    return similarity_score(y_left, reg) + similarity_score(y_right, reg) - similarity_score(y_root, reg)


@jit(nopython=True)
def mean_squared_error(y_true, y_pred):
    """ Returns the mean squared error between y_true and y_pred """
    mse = np.mean(np.power(y_true - y_pred, 2))
    return mse


@jit(nopython=True)
def calculate_variance(X):
    """ Return the variance of the features in dataset X """
    # mean = np.ones(np.shape(X)) * X.mean(0)
    # n_samples = np.shape(X)[0]
    # variance = (1 / n_samples) * np.diag((X - mean).T.dot(X - mean))
    # return variance
    var = np.sum((X - X.mean()) ** 2) / len(X)
    return var


def calculate_std_dev(X):
    """ Calculate the standard deviations of the features in dataset X """
    std_dev = np.sqrt(calculate_variance(X))
    return std_dev


def euclidean_distance(x1, x2):
    """ Calculates the l2 distance between two vectors """
    distance = 0
    # Squared distance between each coordinate
    for i in range(len(x1)):
        distance += pow((x1[i] - x2[i]), 2)
    return math.sqrt(distance)


def accuracy_score(y_true, y_pred):
    """ Compare y_true to y_pred and return the accuracy """
    accuracy = np.sum(y_true == y_pred, axis=0) / len(y_true)
    return accuracy


def calculate_covariance_matrix(X, Y=None):
    """ Calculate the covariance matrix for the dataset X """
    if Y is None:
        Y = X
    n_samples = np.shape(X)[0]
    covariance_matrix = (1 / (n_samples - 1)) * (X - X.mean(axis=0)).T.dot(Y - Y.mean(axis=0))

    return np.array(covariance_matrix, dtype=float)


def calculate_covariance(X, y):
    return np.sum((X - X.mean()) * (y - y.mean())) / len(X)


@jit(nopython=True)
def calculate_dispersion(X):
    return np.sum((X - X.mean()) ** 2)


def calculate_correlation(X, y):
    return calculate_covariance(X, y) / np.sqrt(calculate_variance(X) * calculate_variance(y))


def calculate_correlation_matrix(X, Y=None):
    """ Calculate the correlation matrix for the dataset X """
    if Y is None:
        Y = X
    n_samples = np.shape(X)[0]
    covariance = (1 / n_samples) * (X - X.mean(0)).T.dot(Y - Y.mean(0))
    std_dev_X = np.expand_dims(calculate_std_dev(X), 1)
    std_dev_y = np.expand_dims(calculate_std_dev(Y), 1)
    correlation_matrix = np.divide(covariance, std_dev_X.dot(std_dev_y.T))

    return np.array(correlation_matrix, dtype=float)
