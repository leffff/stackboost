from __future__ import division, print_function
import numpy as np

from stackboost.utils.data_manipulation import divide_on_feature, train_test_split, standardize, to_array
from stackboost.utils.data_operation import calculate_entropy, mean_squared_error, similarity_score, \
    calculate_dispersion, similarity_gain, approximate_quantile_sketch
from stackboost.categorical_encoding import StackedEncoder
from stackboost.utils.data_manipulation import train_test_split, standardize, to_categorical, normalize, target_split
from sklearn.metrics import mean_squared_error, accuracy_score
from stackboost.loss_functions import LogisticLoss, MSE
import torch
import pandas as pd
import warnings
import copy
import seaborn as sns
import matplotlib.pyplot as plt

from functools import lru_cache

warnings.filterwarnings("ignore")
SUPPORTED_ARRAYS = [np.ndarray, torch.tensor, pd.Series, pd.DataFrame, list]


class DecisionNode:
    def __init__(self, feature_i=None, threshold=None, value=None, target=None, true_branch=None,
                 false_branch=None, importance=None):
        self.feature_i = feature_i  # Index for the feature that is tested
        self.threshold = threshold  # Threshold value for feature
        self.value = value  # Value if the node is a leaf in the tree
        self.target = target
        self.true_branch = true_branch  # 'Left' subtree
        self.false_branch = false_branch  # 'Right' subtree
        self.importance = importance

    def is_leaf(self):
        return not self.value is None

    def __str__(self):
        print(
            self.feature_i,
            self.threshold,
            self.value,
            self.target,
            self.true_branch,
            self.false_branch,
            self.importance
        )


# Super class of RegressionTree and ClassificationTree
class DepthwiseTree(object):
    """Super class of RegressionTree and ClassificationTree.
    Parameters:
    -----------
    min_samples_split: int
        The minimum number of samples needed to make a split when building a tree.
    min_impurity: float
        The minimum impurity required to split the tree further.
    max_depth: int
        The maximum depth of a tree.
    loss: function
        Loss function that is used for Gradient Boosting models to calculate impurity.
    """

    def __init__(self, loss_function=LogisticLoss(), min_samples_split: int = 2, min_impurity: float = 1e-7,
                 max_depth: int = float("inf"), l2_leaf_reg: float = 0.0, n_quantiles: int = 33,
                 max_leaf_samples: int = 1000, prune: bool = False, gamma: int = 0):

        self.root = None  # Root node in dec. tree
        # Minimum n of samples to justify split
        self.min_samples_split = min_samples_split
        # The minimum impurity to justify split
        self.min_impurity = min_impurity
        # The maximum depth to grow the tree to
        self.max_depth = max_depth
        # Function to calculate impurity (classif.=>info gain, regr=>variance reduct.)
        self._impurity_calculation = None
        # Function to determine prediction of y at leaf
        self._leaf_value_calculation = None
        # If y is one-hot encoded (multi-dim) or not (one-dim)
        self.one_dim = None
        # If Gradient Boost
        self.loss_function = loss_function
        self.l2_leaf_reg = l2_leaf_reg
        self.n_quantiles = n_quantiles
        self.max_leaf_samples = max_leaf_samples
        self.cat_features = None
        self.sample_weight = None
        self.prune = prune
        self.gamma = gamma

    def fit(self, X, y, cat_features, sample_weight):
        """ Build decision tree """
        self.one_dim = len(np.shape(y)) == 1

        X, y = to_array(X), to_array(y)

        self.cat_features = cat_features

        self.sample_weight = sample_weight
        self.n_samples, self.n_features = X.shape
        if sample_weight is None:
            self.sample_weight = 1 / np.array([1 for i in range(self.n_features)])

        self.root = self._build_tree(X, y)

        if self.prune:
            self.root = self.__prune()

    def _build_tree(self, X, y, current_depth=0):
        """ Recursive method which builds out the decision tree and splits X and respective y
        on the feature of X which (based on impurity) best separates the data"""

        largest_impurity = 0
        best_criteria = None  # Feature index and threshold
        best_sets = None  # Subsets of the data

        # Check if expansion of y is needed
        if len(np.shape(y)) == 1:
            y = np.expand_dims(y, axis=1)

        # Add y as last column of X
        Xy = np.concatenate((X, y), axis=1)

        n_samples, n_features = np.shape(X)

        if n_samples >= self.min_samples_split and current_depth <= self.max_depth:
            # Calculate the impurity for each feature
            for feature_i in range(n_features):
                # All values of feature_i

                feature_values = np.expand_dims(X[:, feature_i], axis=1)
                if len(np.unique(feature_values)) > self.n_quantiles:
                    delta = np.max(feature_values) - np.min(feature_values)
                    quintile_size = delta / self.n_quantiles
                    quintiles = np.array([quintile_size * (i + 1) for i in range(self.n_quantiles)])
                else:
                    quintiles = np.unique(feature_values)

                # w = approximate_quantile_sketch(X[:, feature_i], y[:, :3], y[:, 3:])

                # Iterate through all unique values of feature column i and
                # calculate the impurity
                # sns.distplot(X[feature_i])
                # plt.show()

                for threshold in quintiles:
                    # Divide X and y depending on if the feature value of X at index feature_i
                    # meets the threshold
                    Xy1, Xy2 = divide_on_feature(Xy, feature_i, threshold)

                    if len(Xy1) > 0 and len(Xy2) > 0:
                        # Select the y-values of the two sets
                        y1 = Xy1[:, n_features:]
                        y2 = Xy2[:, n_features:]

                        # Calculate impurity
                        impurity = self._impurity_calculation(y, y1, y2) * self.sample_weight[feature_i]

                        # If this threshold resulted in a higher information gain than previously
                        # recorded save the threshold value and the feature
                        # index
                        if impurity > largest_impurity:
                            largest_impurity = impurity
                            best_criteria = {"feature_i": feature_i, "threshold": threshold, "target": y,
                                             "impurity": impurity}
                            best_sets = {
                                "leftX": Xy1[:, :n_features],  # X of left subtree
                                "lefty": Xy1[:, n_features:],  # y of left subtree
                                "rightX": Xy2[:, :n_features],  # X of right subtree
                                "righty": Xy2[:, n_features:]  # y of right subtree
                            }

        if largest_impurity > self.min_impurity:
            # Build subtrees for the right and left branches
            true_branch = self._build_tree(best_sets["leftX"], best_sets["lefty"], current_depth + 1)
            false_branch = self._build_tree(best_sets["rightX"], best_sets["righty"], current_depth + 1)
            return DecisionNode(feature_i=best_criteria["feature_i"], threshold=best_criteria[
                "threshold"], true_branch=true_branch, false_branch=false_branch,
                                target=best_criteria["target"],
                                importance=(len(y) / self.n_samples) * best_criteria["impurity"])

        # We're at leaf => determine value
        leaf_value = self._leaf_value_calculation(y)

        return DecisionNode(value=leaf_value, target=y)

    def __prune(self, tree=None):
        if tree is None:
            tree = self.root

        # if node is not a leaf
        if tree.true_branch != None:
            tree.true_branch = self.__prune(tree.true_branch)
        if tree.false_branch != None:
            tree.false_branch = self.__prune(tree.false_branch)

        if not tree.is_leaf() and tree.true_branch.is_leaf() and tree.false_branch.is_leaf():
            # print(self._impurity_calculation(tree.target, tree.true_branch.target, tree.false_branch.target))
            if self._impurity_calculation(tree.target, tree.true_branch.target, tree.false_branch.target) - self.gamma < 0:
                print("PRUNED")
                return DecisionNode(value=self._leaf_value_calculation(tree.target), target=tree.target)

        return tree

    def __calculate_importance(self, tree=None):
        if tree is None:
            tree = self.root
            if tree.true_branch == None and tree.false_branch == None:
                return

        # print(tree)
        self.feature_importance[tree.feature_i] += tree.importance

        if tree.true_branch != None:
            tree.true_branch = self.__prune(tree.true_branch)
        if tree.false_branch != None:
            tree.false_branch = self.__prune(tree.false_branch)

    def get_feature_importance(self):
        self.feature_importance = {i: 1 / self.n_features for i in range(self.n_features)}
        self.__calculate_importance()
        return np.array(list(self.feature_importance.values()))

    def predict_value(self, x, tree=None):
        """ Do a recursive search down the tree and make a prediction of the data sample by the
            value of the leaf that we end up at """

        if tree is None:
            tree = self.root

        # If we have a value (i.e we're at a leaf) => return value as the prediction
        if tree.value is not None:
            return tree.value

        # Choose the feature that we will test
        feature_value = x[tree.feature_i]

        # Determine if we will follow left or right branch
        branch = tree.false_branch
        if isinstance(feature_value, int) or isinstance(feature_value, float):
            if feature_value >= tree.threshold:
                branch = tree.true_branch
        elif feature_value == tree.threshold:
            branch = tree.true_branch

        # Test subtree
        return self.predict_value(x, branch)

    def predict(self, X):
        """ Classify samples one by one and return the set of labels """
        y_pred = [self.predict_value(sample) for sample in X]
        return y_pred


class SGBoostTree(DepthwiseTree):
    def _split(self, y):
        col = int(np.shape(y)[1] / 2)
        y, y_pred = y[:, :col], y[:, col:]
        return y, y_pred

    def _gain(self, y, y_pred):
        nominator = np.power((y * self.loss_function.gradient(y, y_pred)).sum(), 2)
        denominator = self.loss_function.hess(y, y_pred).sum()
        return 0.5 * (nominator / denominator)

    def _gain_by_taylor(self, y, y1, y2):
        # Split
        y, y_pred = self._split(y)
        y1, y1_pred = self._split(y1)
        y2, y2_pred = self._split(y2)

        true_gain = self._gain(y1, y1_pred)
        false_gain = self._gain(y2, y2_pred)
        gain = self._gain(y, y_pred)
        return true_gain + false_gain - gain

    def _approximate_update(self, y):
        # y split into y, y_pred
        y, y_pred = self._split(y)
        # Newton's Method
        gradient = np.sum(y * self.loss_function.gradient(y, y_pred), axis=0)
        hessian = np.sum(self.loss_function.hess(y, y_pred), axis=0)
        update_approximation = gradient / (hessian + self.l2_leaf_reg)
        # print(update_approximation)
        return update_approximation

    def fit(self, X, y, sample_weight=None, cat_features=None):
        self._impurity_calculation = self._gain_by_taylor
        self._leaf_value_calculation = self._approximate_update
        super(SGBoostTree, self).fit(X, y, sample_weight, cat_features)


class SGBoost(object):
    def __init__(self, n_estimators=100, learning_rate=0.001, min_samples_split=2,
                 min_impurity=1e-7, max_depth=2, loss_function=LogisticLoss(), l2_leaf_reg: float = 0.0,
                 n_quantiles: int = 33,
                 max_leaf_samples: int = 1000, prune: bool = False, gamma: int = 0, metrics: list = [],
                 verbose: bool = False, regression=False):

        self.n_estimators = n_estimators  # Number of trees
        self.learning_rate = learning_rate  # Step size for weight update
        self.min_samples_split = min_samples_split  # The minimum n of sampels to justify split
        self.min_impurity = min_impurity  # Minimum variance reduction to continue
        self.max_depth = max_depth  # Maximum depth for tree
        self.l2_leaf_reg = l2_leaf_reg
        self.n_quantiles = n_quantiles
        self.max_leaf_samples = max_leaf_samples
        self.prune = prune
        self.gamma = gamma
        self.metrics = metrics
        self.verbose = verbose
        self.regression = regression

        # Log loss for classification
        self.loss_function = loss_function

        # Initialize regression trees
        self.trees = []
        for _ in range(n_estimators):
            tree = SGBoostTree(
                min_samples_split=self.min_samples_split,
                min_impurity=min_impurity,
                max_depth=self.max_depth,
                loss_function=self.loss_function,
                l2_leaf_reg=self.l2_leaf_reg,
                n_quantiles=self.n_quantiles,
                max_leaf_samples=self.max_leaf_samples,
                prune=self.prune,
                gamma=self.gamma
            )

            self.trees.append(tree)

    def fit(self, X, y, eval_set=None, use_best_model=None):
        y_pred = np.full(np.shape(y), y.mean())

        sample_weights = np.full((X.shape[0]), 1 / X.shape[1])

        for i in range(self.n_estimators):
            min_error = float('inf')

            tree = self.trees[i]
            y_and_pred = np.concatenate((y, y_pred), axis=1)
            tree.fit(X, y_and_pred)
            # sample_weights = tree.get_feature_importance()
            update_pred = tree.predict(X)

            y_pred -= np.multiply(self.learning_rate, update_pred)

            if self.verbose:
                train_scoring_output = ""
                test_scoring_output = ""

                if self.metrics != []:
                    train_scores = [scoring_function(np.argmax(y, axis=1), np.argmax(y_pred, axis=1)) for scoring_function in self.metrics]
                    train_scoring_output = f"train metrics: {train_scores}"
                    if not eval_set is None:
                        test_scores = [scoring_function(eval_set[1], np.argmax(self.__inner_predict(eval_set[0], i), axis=1)) for scoring_function in self.metrics]
                        test_scoring_output = f"test metrics: {test_scores}"

                print(f"Tree: {i}", train_scoring_output, test_scoring_output)

    def predict(self, X):
        return self.__inner_predict(X, self.n_estimators)


    def __inner_predict(self, X, iteration):
        y_pred = None
        # Make predictions
        for tree in self.trees[:iteration + 1]:
            # Estimate gradient and update prediction
            update_pred = tree.predict(X)
            if y_pred is None:
                y_pred = np.zeros_like(update_pred)
            y_pred -= np.multiply(self.learning_rate, update_pred)

        return y_pred


class SGBRegressor(SGBoost):
    def __init__(self, n_estimators=100, learning_rate=0.001, min_samples_split=2,
                 min_impurity=1e-7, max_depth=2, loss_function=MSE(), l2_leaf_reg: float = 0.0,
                 n_quantiles: int = 33,
                 max_leaf_samples: int = 1000, prune: bool = False, gamma: int = 0, metrics: list = [],
                 verbose: bool = False):
        super(SGBRegressor, self).__init__(n_estimators=n_estimators,
                                           learning_rate=learning_rate,
                                           min_samples_split=min_samples_split,
                                           min_impurity=min_impurity,
                                           max_depth=max_depth,
                                           loss_function=loss_function,
                                           l2_leaf_reg=l2_leaf_reg,
                                           n_quantiles=n_quantiles,
                                           max_leaf_samples=max_leaf_samples,
                                           prune=prune,
                                           gamma=gamma,
                                           metrics=metrics,
                                           verbose=verbose,
                                           regression=True)

    def fit(self, X, y, eval_set=None, use_best_model=None):
        y = y.reshape(-1, 1)
        super(SGBRegressor, self).fit(X, y, eval_set, use_best_model)


class SGBClassifier(SGBoost):
    def __init__(self, n_estimators=100, learning_rate=0.001, min_samples_split=2,
                 min_impurity=1e-7, max_depth=2, loss_function=LogisticLoss(), l2_leaf_reg: float = 0.0,
                 n_quantiles: int = 33,
                 max_leaf_samples: int = 1000, prune: bool = False, gamma: int = 0, metrics: list = [],
                 verbose: bool = False):
        super(SGBClassifier, self).__init__(n_estimators=n_estimators,
                                            learning_rate=learning_rate,
                                            min_samples_split=min_samples_split,
                                            min_impurity=min_impurity,
                                            max_depth=max_depth,
                                            loss_function=loss_function,
                                            l2_leaf_reg=l2_leaf_reg,
                                            n_quantiles=n_quantiles,
                                            max_leaf_samples=max_leaf_samples,
                                            prune=prune,
                                            gamma=gamma,
                                            metrics=metrics,
                                            verbose=verbose,
                                            regression=False)

    def fit(self, X, y, eval_set=None, use_best_model=None):
        y = to_categorical(y)
        super(SGBClassifier, self).fit(X, y, eval_set, use_best_model)

    def predict_proba(self, X):
        y_pred = super(SGBClassifier, self).predict(X)
        y_pred = np.exp(y_pred) / np.expand_dims(np.sum(np.exp(y_pred), axis=1), axis=1)
        return y_pred

    def predict(self, X):
        y_pred = self.predict_proba(X)
        y_pred = np.argmax(y_pred, axis=1)
        return y_pred
