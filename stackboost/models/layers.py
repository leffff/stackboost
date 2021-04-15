import numpy as np
from waveml import WaveTransformer, WaveStackingTransformer
from stackboost.models.tree import ErrorTreeRegressor, SimilarityTreeRegressor, DispersionTreeRegressor
from sklearn.metrics import mean_squared_error
from numba.experimental import jitclass
from numba import float32


class ModelLayer:
    def __init__(self, estimators, n_folds: int = 6, random_state: int = None, shuffle: bool = False,
                 verbose: bool = True, metric=mean_squared_error):

        self.estimators = estimators
        self.n_folds = n_folds
        self.metric = metric
        self.random_state = random_state
        self.shuffle = shuffle
        self.verbose = verbose

        self.stack = WaveStackingTransformer(models=self.estimators, n_folds=self.n_folds,
                                             random_state=self.random_state,
                                             shuffle=self.shuffle, metric=self.metric, verbose=self.verbose)
        self.tuner = WaveTransformer()

    def fit(self, X, y):
        XS = self.stack.fit_transform(X, y)
        # self.tuner.fit(XS, y)

    def predict(self, X):
        XS = self.stack.transform(X)
        # XT = self.tuner.transform(XS)
        # np.hstack([XS, XT])
        return XS


spec = [
    ("dropout_percentage", float32)
]


@jitclass(spec=spec)
class DropoutLayer:
    def __init__(self, dropout_percentage):
        self.dropout_percentage = dropout_percentage

    def fit(self, X, y):
        pass

    def predict(self, X):
        n_cols = X.shape[1]
        return X[:, np.random.randint(low=0, high=n_cols, size=(n_cols - int(self.dropout_percentage * n_cols)))]
