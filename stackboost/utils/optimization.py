import numpy as np
from numba import jit, njit
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans

@njit()
def min_max_sketch(X: np.ndarray, n_quantiles: int) -> np.ndarray:
    if len(np.unique(X)) > n_quantiles:
        delta = np.max(X) - np.min(X)
        quintile_size = delta / n_quantiles
        quintiles = np.array([quintile_size * (i + 1) for i in range(n_quantiles)])
    else:
        quintiles = np.unique(X)

    return quintiles


@njit()
def hist_sketch(X: np.ndarray, n_quantiles: int) -> np.ndarray:
    quintiles = np.histogram(a=X, bins=n_quantiles)[1]
    return quintiles


@njit()
def greedy_sketch(X: np.ndarray, n_quantiles: int) -> np.ndarray:
    return X


def cluster_sketch(X: np.ndarray, n_quantiles: int) -> np.ndarray:
    model = KMeans(n_clusters=n_quantiles)
    classes = model.fit_predict(X)
    clusters = [X[classes == i] for i in range(n_quantiles)]
    quintiles = np.array([(clusters[i].max() + clusters[i + 1].min()) / 2 for i in range(n_quantiles - 1)])
    return quintiles


def make_bins(X: np.ndarray, n_bins: int) -> np.ndarray:
    label_encoder =  LabelEncoder()
    for feature in range(X.shape[1]):
        X[:, feature] = label_encoder.fit_transform(pd.cut(X[:, feature], n_bins, retbins=True)[0])
    return X


def undersampling(gradients: np.ndarray, undersampling_percentage: float) -> np.ndarray:
    flat_gradients = gradients.flatten()
    n_samples = flat_gradients.shape[0]
    absolute_gradients = np.abs(flat_gradients)
    indexes = absolute_gradients.argsort()[round(n_samples * undersampling_percentage):]
    mask = np.zeros(n_samples)
    mask[indexes] = 1
    return mask == 1
