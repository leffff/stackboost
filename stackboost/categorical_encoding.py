from stackboost.utils.data_manipulation import to_array
import numpy as np
from numba import float32, boolean, int32, char
from numba.core.types.npytypes import Array
from numba.core.types.misc import UnicodeType
from numba.experimental import jitclass
import torch
import pandas as pd
import numba
from stackboost.utils.errors import NotFittedError

SUPPORTED_ARRAYS = [np.ndarray, torch.tensor, pd.Series, pd.DataFrame, list]


class StackedEncoder:
    """
    Categorical feature incoding model
    """
    def __init__(self, strategy: str = "mean", regression: bool = True):
        self.strategy = strategy
        if self.strategy != None:
            self.strategies = {"mean": np.mean,
                               "median": np.median,
                               "sum": np.sum}

            self.strategy = self.strategy.lower()
            if self.strategy not in list(self.strategies.keys()):
                raise ValueError("Given strategy type", self.strategy, "allowed," ', '.join(list(self.strategies.keys())))
        self.regression = regression
        self.fitted = False

    def fit(self, X, y, cat_features=None) -> None:
        self.fitted = False
        X = to_array(X)
        y = to_array(y)

        self.n_features = X.shape[1]

        self.cat_features = cat_features
        if self.cat_features == None:
            self.cat_features = np.array([i for i in range(self.n_features)])

        self.__target(X, y)

    def __target(self, X: np.ndarray, y: np.ndarray) -> None:
        if self.regression:
            self.mappers = np.array([])
            for i in range(len(self.cat_features)):
                counts = dict()
                feature = X[:, self.cat_features[i]]
                uniques = np.unique(feature)

                for unique in uniques:
                    counts[unique] = self.strategies.get(self.strategy)(y[feature == unique])

                self.mappers = np.append(self.mappers, counts)
        else:
            self.mappers = np.array([])
            for i in range(len(self.cat_features)):
                counts = dict()
                feature = X[:, self.cat_features[i]]
                uniques = np.unique(feature)

                for unique in uniques:
                    counts[unique] = y[feature == unique].mean()

                self.mappers = np.append(self.mappers, counts)

    def transform(self, X: np.ndarray) -> np.ndarray:
        X_array = to_array(X)
        n_features_test = X_array.shape[1]
        if self.n_features != n_features_test:
            raise ValueError(
                f"Shape of fitting array ({self.n_features} columns) and transforming array ({n_features_test} columns) do not match")

        for i in range(len(self.cat_features)):
            X_array[:, self.cat_features[i]] = np.vectorize(self.mappers[i].get)(X_array[:, self.cat_features[i]])

        return X_array

    def fit_transform(self, X, y=None, regression=True, cat_features=None) -> np.ndarray:
        self.fit(X, y, regression, cat_features)
        return self.transform(X)
