import numpy as np
from sklearn.metrics import accuracy_score
from numba.experimental import jitclass
from stackboost.utils.activation_functions import Sigmoid


class Loss(object):
    def loss(self, y_true, y_pred):
        return NotImplementedError()

    def gradient(self, y, y_pred):
        raise NotImplementedError()

    def acc(self, y, y_pred):
        return 0


@jitclass()
class SquareLoss(Loss):
    def __init__(self): pass

    def loss(self, y, y_pred):
        return np.mean(0.5 * np.power((y - y_pred), 2))

    def gradient(self, y, y_pred):
        return -(y - y_pred)

    def hess(self, y, y_pred):
        return np.array([1 for i in range(len(y))])


class CrossEntropy(Loss):
    def __init__(self):
        pass

    def loss(self, y, p):
        p = np.clip(p, 1e-15, 1 - 1e-15)
        return np.mean(- y * np.log(p) - (1 - y) * np.log(1 - p))

    def acc(self, y, p):
        return accuracy_score(np.argmax(y, axis=1), np.argmax(p, axis=1))

    def gradient(self, y, p):
        p = np.clip(p, 1e-15, 1 - 1e-15)
        return - (y / p) + (1 - y) / (1 - p)


@jitclass()
class MSE(Loss):
    def __init__(self): pass

    def loss(self, y, y_pred):
        return np.mean((y - y_pred) ** 2)

    def gradient(self, y, y_pred):
        return -2 / len(y) * (y - y_pred)

    def hess(self, y, y_pred):
        return np.full(y.shape, 2 / len(y))


class LogisticLoss():
    def __init__(self):
        self.log_func = Sigmoid()

    def loss(self, y, y_pred):
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        p = self.log_func(y_pred)
        return np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))

    # gradient w.r.t y_pred
    def gradient(self, y, y_pred):
        p = self.log_func(y_pred)
        return -(y - p)

    # w.r.t y_pred
    def hess(self, y, y_pred):
        p = self.log_func(y_pred)
        return p * (1 - p)
