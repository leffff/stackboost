import numpy as np


class Softmax:
    def __call__(self, X):
        return np.exp(X) / np.expand_dims(np.sum(np.exp(X), axis=1), axis=1)


class Sigmoid:
    def __call__(self, x):
        return 1 / (1 + np.exp(-x))

    def gradient(self, x):
        return self.__call__(x) * (1 - self.__call__(x))


class Softplus:
    def __call__(self, X):
        return np.log(1 - np.exp(X))
