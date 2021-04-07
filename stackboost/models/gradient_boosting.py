from stackboost.utils.data_manipulation import to_array
from stackboost.loss_functions import MSE, SquareLoss
from stackboost.utils.errors import NotFittedError
from stackboost.categorical_encoding import StackedEncoder
import plotly.graph_objects as go
import torch
import pandas as pd
import numpy as np
from copy import copy

SUPPORTED_ARRAYS = [np.ndarray, torch.tensor, pd.Series, pd.DataFrame, list]


class StackedGradientBoostingRegressor:
    def __init__(self, estimator=None, n_estimators=50, learning_rate=0.01, loss_function=SquareLoss(),
                 verbose: bool = True):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.loss_function = loss_function
        self.model = estimator
        self.verbose = verbose
        self.use_best_model = None
        self.models = None
        self.fitted = False
        self.encoder = None

    def fit(self, X, y, eval_set=None, cat_features=None, use_best_model: bool = True):
        X, y = to_array(X), to_array(y)

        self.test_pres = False
        if eval_set != None:
            self.test_pres = True
            X_test, y_test = eval_set

        self.cat_features = cat_features
        if type(self.cat_features) in SUPPORTED_ARRAYS:
            print("Encoded")
            self.encoder = StackedEncoder()
            X = self.encoder.fit_transform(X, y, cat_features=self.cat_features)

        self.use_best_model = use_best_model

        self.initial_preds = np.array([y.mean() for i in range(len(y))])
        residuals = self.loss_function.gradient(y, self.initial_preds)

        self.models = []
        self.train_errors = np.array([])
        self.test_errors = np.array([])
        predictions = [self.initial_preds]
        pseudo_residuals = [residuals]

        for i in range(self.n_estimators):
            model = copy(self.model)
            model.fit(X, pseudo_residuals[-1])
            self.models.append(model)

            preds = model.predict(X) * self.learning_rate + predictions[-1]
            predictions.append(preds)

            train_error = self.loss_function.loss(y, preds)
            self.train_errors = np.append(self.train_errors, train_error)

            if self.test_pres:
                test_error = self.loss_function.loss(y_test, self.__inner_predict(X_test))
                self.test_errors = np.append(self.test_errors, test_error)

            if self.verbose:
                train_print = f"Estimator: {i} Train: {train_error}"
                test_print = ""
                if self.test_pres:
                    test_print = f" Test: {test_error}"
                print(train_print + test_print)

            residuals -= self.loss_function.gradient(y, preds)
            pseudo_residuals.append(residuals)

        self.fitted = True

    def __inner_predict(self, X):
        X = to_array(X)
        y_pred = np.array([self.initial_preds[0] for i in range(len(X))])
        for i in range(len(self.models)):
            y_pred += self.models[i].predict(X) * self.learning_rate
        return y_pred

    def plot(self):
        if not self.fitted:
            raise NotFittedError()

        iterations = [i for i in range(self.n_estimators)]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=iterations, y=self.train_errors,
                                 mode='lines',
                                 name='train'))
        fig.add_trace(go.Scatter(x=iterations, y=self.test_errors,
                                 mode='lines',
                                 name='test'))
        fig.show()

    def predict(self, X):
        if not self.fitted:
            raise NotFittedError()

        if type(self.cat_features) in SUPPORTED_ARRAYS and self.encoder != None:
            X = self.encoder.transform(X)

        X = to_array(X)
        y_pred = np.array([self.initial_preds[0] for i in range(len(X))])

        if self.use_best_model:
            if self.test_pres:
                models = self.models[:np.argmin(self.test_errors)]
            else:
                models = self.models[:np.argmin(self.train_errors)]
        else:
            models = self.models

        for i in range(len(models)):
            y_pred += self.models[i].predict(X) * self.learning_rate

        return y_pred
