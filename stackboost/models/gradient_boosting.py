from stackboost.utils.data_manipulation import to_array
from stackboost.models.tree_test import StackedTreeRegressor
from stackboost.loss_functions import MSE, SquareLoss
from stackboost.utils.errors import NotFittedError
import plotly.graph_objects as go
import numpy as np
from copy import copy


class StackedGradientBoostingRegressor:
    def __init__(self, estimator=None, n_estimators=50, learning_rate=0.01, loss_function=SquareLoss()):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.loss_function = loss_function
        self.model = estimator
        self.models = None
        self.fitted = False

    def fit(self, X, y, eval_set=None, cat_features=None):
        X, y = to_array(X), to_array(y)
        X_test, y_test = eval_set
        self.initial_preds = np.array([y.mean() for i in range(len(y))])
        residuals = self.loss_function.gradient(y, self.initial_preds)

        self.models = []
        self.train_errors = []
        self.test_errors = []
        predictions = [self.initial_preds]
        pseudo_residuals = [residuals]

        for i in range(self.n_estimators):
            print(i)
            model = copy(self.model)
            model.fit(X, pseudo_residuals[-1], cat_features=cat_features)
            self.models.append(model)

            preds = model.predict(X) * self.learning_rate + predictions[-1]
            predictions.append(preds)

            self.train_errors.append(self.loss_function.loss(y, preds))
            self.test_errors.append(self.loss_function.loss(y_test, self.__inner_predict(X_test)))

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

        X = to_array(X)
        return self.__inner_predict(X)
