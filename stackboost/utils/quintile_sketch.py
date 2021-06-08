import numpy as np
from numba import jit


@jit(nopython=True)
def min_max_sketch(X: np.ndarray, n_quantiles: int) -> np.ndarray:
    if len(np.unique(X)) > n_quantiles:
        delta = np.max(X) - np.min(X)
        quintile_size = delta / n_quantiles
        quintiles = np.array([quintile_size * (i + 1) for i in range(n_quantiles)])
    else:
        quintiles = np.unique(X)

    return quintiles


@jit(nopython=True)
def hist_sketch(X: np.ndarray, n_quantiles: int) -> np.ndarray:
    quintiles = np.histogram(a=X, bins=n_quantiles)[1]
    return quintiles
