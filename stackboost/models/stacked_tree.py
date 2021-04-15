from __future__ import division, print_function
import numpy as np

from stackboost.utils.data_manipulation import divide_on_feature, train_test_split, standardize, to_array
from stackboost.utils.data_operation import calculate_entropy, accuracy_score, calculate_variance, \
    calculate_correlation, calculate_correlation_matrix, calculate_covariance, calculate_dispersion, mean_squared_error, \
    similarity_score
from stackboost.categorical_encoding import StackedEncoder
import torch
import pandas as pd
from sklearn.linear_model import LogisticRegression, LinearRegression
import warnings
import copy
from numba import jit, int32, float32
from numba.experimental import jitclass
from stackboost.models.tree import SimilarityTreeRegressor, ErrorTreeRegressor, DispersionTreeRegressor
from sklearn.metrics import mean_squared_error
from copy import copy
from sklearn.linear_model import LinearRegression

warnings.filterwarnings("ignore")
SUPPORTED_ARRAYS = [np.ndarray, torch.tensor, pd.Series, pd.DataFrame, list]


class StackedTreeRegressor:
    def __init__(self, layers: [list, tuple], head=None, min_samples_split: int = 2, min_impurity: float = 1e-7,
                 max_depth: int = float("inf"), l2_leaf_reg: float = 0.0, n_quantiles: int = 33,
                 loss_function=mean_squared_error, metric=mean_squared_error, n_folds: int = 4,
                 random_state: int = None, shuffle: bool = False, extraction_power: float = 0.1, verbose: bool = True):

        self.layers = layers
        self.head = head
        # self.min_samples_split = min_samples_split
        # self.min_impurity = min_impurity
        # self.max_depth = max_depth
        # self.l2_leaf_reg = l2_leaf_reg
        # self.n_quantiles = n_quantiles
        # self.loss_function = loss_function

        self.n_folds = n_folds
        self.metric = metric
        self.random_state = random_state
        self.shuffle = shuffle
        self.verbose = verbose
        self.regression = True

    def fit(self, X, y):
        preds = []
        self.layers[0].fit(X, y)
        preds.append(self.layers[0].predict(X))
        for i in range(1, len(self.layers)):
            self.layers[i].fit(preds[i - 1], y)
        self.head.fit(preds[-1], y)

    def predict(self, X):
        preds = []
        preds.append(self.layers[0].predict(X))
        for i in range(1, len(self.layers)):
            preds.append(self.layers[i].predict(preds[i - 1]))

        return self.head.predict(preds[-1])
