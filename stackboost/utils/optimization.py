import numpy as np
from numba import njit

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


def undersampling(gradients: np.ndarray, undersampling_percentage: float) -> np.ndarray:
    absolute_gradients = np.abs(gradients)
    summed_gradients = absolute_gradients.sum(axis=1)
    n_samples = summed_gradients.shape[0]
    indexes = summed_gradients.argsort()[round(n_samples * undersampling_percentage):]
    mask = np.zeros(n_samples)
    mask[indexes] = 1
    return mask == 1
