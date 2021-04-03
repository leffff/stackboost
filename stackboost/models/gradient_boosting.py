from stackboost.utils.data_manipulation import to_array
from stackboost.models.split_test import StackedTreeRegressor
from stackboost.loss_functions import MSE, SquareLoss
import numpy as np
from copy import copy


class StackedGradientBoostingRegressor:
    def __init__(self, leaf_estimaor=None, n_estimators=50, max_depth=15, learning_rate=0.1, loss_function=SquareLoss()):
        self.leaf_estimator = leaf_estimaor
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.loss_function = loss_function
        self.model = StackedTreeRegressor(leaf_estimator=self.leaf_estimator, max_depth=self.max_depth)
        self.models = None
        self.fitted = False

    def fit(self, X, y, cat_features=None):
        X, y = to_array(X), to_array(y)

        self.initial_preds = np.array([y.mean() for i in range(len(y))])
        residuals = self.loss_function.gradient(y, self.initial_preds)

        self.models = []
        predictions = [self.initial_preds]
        pseudo_residuals = [residuals]

        for i in range(self.n_estimators):
            print(i)
            model = copy(self.model)
            model.fit(X, pseudo_residuals[-1], cat_features=cat_features)
            self.models.append(model)

            preds = model.predict(X) * self.learning_rate + predictions[-1]
            predictions.append(preds)

            residuals -= self.loss_function.gradient(y, preds)
            pseudo_residuals.append(residuals)


    def predict(self, X):
        X = to_array(X)
        y_pred = np.array([self.initial_preds[0] for i in range(len(X))])
        for i in range(self.n_estimators):
            y_pred += self.models[i].predict(X) * self.learning_rate

        return y_pred


