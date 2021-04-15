from .loss_functions import MSE, SquareLoss, CrossEntropy
from .models.tree import SimilarityTreeRegressor, ErrorTreeRegressor, DispersionTreeRegressor
from .models.gradient_boosting import StackedGradientBoostingRegressor
from .models.stacked_tree import StackedTreeRegressor
from .models.layers import ModelLayer, DropoutLayer

__author__ = 'Lev Novitskiy > levnovitskiy@gmail.com > https://github.com/leffff/ > https://www.kaggle.com/levnovitskiy'
__license__ = 'MIT'
