from __future__ import division, print_function
import numpy as np

from stackboost.utils.data_manipulation import divide_on_feature, to_array
from stackboost.utils.quintile_sketch import min_max_sketch, hist_sketch
from stackboost.utils.data_manipulation import to_categorical
from stackboost.utils.loss_functions import LogisticLoss, SquareLoss
import torch
import pandas as pd
import warnings
from stackboost.utils.activation_functions import Sigmoid

warnings.filterwarnings("ignore")
SUPPORTED_ARRAYS = [np.ndarray, torch.tensor, pd.Series, pd.DataFrame, list]


class DecisionNode:
    def __init__(self, feature_i=None, threshold=None, value=None, target=None, true_branch=None,
                 false_branch=None, importance=None, depth=None):
        self.feature_i = feature_i
        self.threshold = threshold
        self.value = value
        self.target = target
        self.true_branch = true_branch
        self.false_branch = false_branch
        self.importance = importance
        self.depth = depth

    def is_leaf(self):
        return not self.value is None

    def __str__(self):
        return f"feature_i: {self.feature_i}\n" \
               f"threshold: {self.threshold}\n" \
               f"value: {self.value}\n" \
               f"true_branch: {self.true_branch}\n" \
               f"false_branch: {self.false_branch}\n" \
               f"importance: {self.importance}\n" \
               f"depth: {self.depth}\n" \
               f"is_leaf: {self.is_leaf()}"


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
                 max_leaf_samples: int = 1000, prune: bool = False, gamma_percentage: float = 1.0, position=None,
                 speedup: bool = False):

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
        self.sample_weight = None
        self.prune = prune
        self.gamma_percentage = gamma_percentage
        self.position = position
        self.speedup = speedup

        self.current_depth = 0
        self.n_nodes = 1
        self.total_gain = 0
        self.average_gain = 0

    def fit(self, X, y, sample_weight):
        """ Build decision tree """
        self.one_dim = len(np.shape(y)) == 1
        self.n_samples, self.n_features = X.shape

        self.sample_weight = sample_weight

        self.root = self._build_tree(X, y)
        self.average_gain = self.total_gain / self.n_nodes

        if self.prune:
            self.root = self.__prune()

    def _build_tree(self, X, y, current_depth=0):
        """ Recursive method which builds out the decision tree and splits X and respective y
        on the feature of X which (based on impurity) best separates the data"""

        if current_depth > self.current_depth:
            self.current_depth = current_depth

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
                quintiles = min_max_sketch(X=feature_values, n_quantiles=self.n_quantiles)

                # print(quintiles)
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
                            best_criteria = {"feature_i": feature_i,
                                             "threshold": threshold,
                                             "target": y,
                                             "impurity": impurity,
                                             "depth": current_depth}
                            best_sets = {
                                "leftX": Xy1[:, :n_features],  # X of left subtree
                                "lefty": Xy1[:, n_features:],  # y of left subtree
                                "rightX": Xy2[:, :n_features],  # X of right subtree
                                "righty": Xy2[:, n_features:]  # y of right subtree
                            }
                            self.total_gain += impurity

        if largest_impurity > self.min_impurity:
            # Build subtrees for the right and left branches
            true_branch = self._build_tree(best_sets["leftX"], best_sets["lefty"], current_depth + 1)
            false_branch = self._build_tree(best_sets["rightX"], best_sets["righty"], current_depth + 1)
            self.n_nodes += 1
            return DecisionNode(
                feature_i=best_criteria["feature_i"],
                threshold=best_criteria["threshold"],
                true_branch=true_branch,
                false_branch=false_branch,
                target=best_criteria["target"],
                importance=(len(y) / self.n_samples) * best_criteria["impurity"],
                depth=best_criteria["depth"]
            )

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
            if self._impurity_calculation(tree.target, tree.true_branch.target,
                                          tree.false_branch.target) - self.average_gain * self.gamma_percentage < 0:
                return DecisionNode(value=self._leaf_value_calculation(tree.target), target=tree.target)

        return tree

    def __calculate_importance(self, importance, tree=None) -> np.ndarray:
        # if tree is none, than we treat it as a root
        if tree is None:
            tree = self.root
        # if both tree branches are None than the node ia s leaf
        if tree.is_leaf():
            return importance

        importance[tree.feature_i] += tree.importance

        return self.__calculate_importance(importance, tree=tree.true_branch) + \
               self.__calculate_importance(importance, tree=tree.false_branch)

    def get_feature_importance(self) -> np.ndarray:
        feature_importance = self.__calculate_importance(self.sample_weight)
        return feature_importance

    def predict_value(self, x, tree=None) -> np.ndarray:
        """ Do a recursive search down the tree and make a prediction of the data sample by the
            value of the leaf that we end up at """

        if tree is None:
            tree = self.root

        # If we have a value (i.e we're at a leaf) => return value as the prediction
        if tree.is_leaf():
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
        # Classify samples one by one and return the set of labels
        y_pred = np.vstack([self.predict_value(sample) for sample in X])
        return y_pred


class StackBoostTree(DepthwiseTree):
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
        # calculate gain
        true_gain = self._gain(y1, y1_pred)
        false_gain = self._gain(y2, y2_pred)
        gain = self._gain(y, y_pred)
        return true_gain + false_gain - gain

    def _approximate_update(self, y):
        # y split into y, y_pred
        y, y_pred = self._split(y)
        # Newton's Method
        if self.speedup:
            speedup_multiplier = (self.position + 1)
        else:
            speedup_multiplier = 1

        gradient = np.sum(y * speedup_multiplier * self.loss_function.gradient(y, y_pred), axis=0)
        hessian = np.sum(self.loss_function.hess(y, y_pred), axis=0)
        update_approximation = gradient / (hessian + self.l2_leaf_reg)

        return update_approximation

    def fit(self, X, y, sample_weight):
        self._impurity_calculation = self._gain_by_taylor
        self._leaf_value_calculation = self._approximate_update
        super(StackBoostTree, self).fit(X, y, sample_weight)


class StackBoost(object):
    def __init__(self, n_estimators=100, learning_rate=0.001, min_samples_split=2,
                 min_impurity=1e-7, max_depth=2, loss_function=LogisticLoss(), l2_leaf_reg: float = 0.0,
                 n_quantiles: int = 33, max_leaf_samples: int = 1000, prune: bool = False,
                 gamma_percentage: float = 1.0, n_folds=4,
                 metrics: list = [], verbose: bool = False, regression=False, speedup: bool = False):

        self.n_estimators = n_estimators  # Number of trees
        self.learning_rate = learning_rate  # Step size for weight update
        self.min_samples_split = min_samples_split  # The minimum n of sampels to justify split
        self.min_impurity = min_impurity  # Minimum variance reduction to continue
        self.max_depth = max_depth  # Maximum depth for tree
        self.l2_leaf_reg = l2_leaf_reg
        self.n_quantiles = n_quantiles
        self.max_leaf_samples = max_leaf_samples
        self.prune = prune
        self.gamma_percentage = gamma_percentage
        self.n_folds = n_folds
        self.metrics = metrics
        self.verbose = verbose
        self.regression = regression

        self.sample_weight = None
        self.cat_features = None
        self.eval_set = None
        self.use_best_model = None
        self.test_errors = None
        self.best_iteration = None

        # Log loss for classification
        self.loss_function = loss_function

        # Initialize regression trees
        self.trees = []
        for _ in range(n_estimators):
            tree = StackBoostTree(
                min_samples_split=self.min_samples_split,
                min_impurity=min_impurity,
                max_depth=self.max_depth,
                loss_function=self.loss_function,
                l2_leaf_reg=self.l2_leaf_reg,
                n_quantiles=self.n_quantiles,
                max_leaf_samples=self.max_leaf_samples,
                prune=self.prune,
                gamma_percentage=self.gamma_percentage,
                position=_,
                speedup=speedup
            )

            self.trees.append(tree)

    def fit(self, X, y, cat_features, sample_weight, eval_set, use_best_model):
        X, y = to_array(X), to_array(y)
        self.n_samples, self.n_features = X.shape

        self.sample_weight = sample_weight
        if self.sample_weight is None:
            self.sample_weight = 1 / np.array([1 for i in range(self.n_features)])

        self.cat_features = cat_features
        if cat_features is None:
            self.cat_features = []

        self.eval_set = eval_set
        self.use_best_model = use_best_model

        self.initial_prediction = y.mean() if self.regression else float(np.argmax(np.mean(y, axis=0)))
        y_pred = np.full(np.shape(y), self.initial_prediction)

        self.test_errors = []

        for i in range(self.n_estimators):
            tree = self.trees[i]
            y_and_pred = np.concatenate((y, y_pred), axis=1)
            tree.fit(X, y_and_pred, sample_weight=self.sample_weight)
            update_pred = tree.predict(X)
            y_pred -= np.multiply(self.learning_rate, update_pred)

            if self.verbose:
                train_scoring_output = ""
                test_scoring_output = ""

                if self.metrics != []:
                    if self.regression:
                        train_scores = [scoring_function(y, y_pred) for scoring_function in self.metrics]
                    else:
                        train_scores = [scoring_function(np.argmax(y, axis=1), np.argmax(y_pred, axis=1)) for
                                        scoring_function in self.metrics]

                    train_scoring_output = f"train metrics: {train_scores}"
                    if not eval_set is None:
                        if self.regression:
                            test_scores = [scoring_function(eval_set[1], self.__inner_predict(eval_set[0], i)) for
                                           scoring_function in self.metrics]

                            # Fix:
                            self.test_errors.append(
                                self.loss_function.loss(eval_set[1], self.__inner_predict(eval_set[0], i)))

                        else:
                            test_scores = [
                                scoring_function(eval_set[1], np.argmax(self.__inner_predict(eval_set[0], i), axis=1))
                                for scoring_function in self.metrics]

                            # Fix:
                            self.test_errors.append(self.loss_function.loss(eval_set[1], np.argmax(
                                self.__inner_predict(eval_set[0], i), axis=1)))

                        test_scoring_output = f"test metrics: {test_scores}"

                print(f"Tree: {i}", train_scoring_output, test_scoring_output)

        # Fix:

        if self.use_best_model:
            print(self.test_errors)
            self.best_iteration = np.argmin(self.test_errors)
        else:
            self.best_iteration = self.n_estimators

    def predict(self, X):
        return self.__inner_predict(X, self.best_iteration)

    def __inner_predict(self, X, iteration):
        y_pred = np.full(np.shape(X)[0], self.initial_prediction).reshape(-1, 1) if self.regression else None

        # Make predictions
        for tree in self.trees[:iteration + 1]:
            # Estimate gradient and update prediction
            update_pred = tree.predict(X)
            if y_pred is None:
                y_pred = np.zeros_like(update_pred)
            y_pred -= np.multiply(self.learning_rate, update_pred)

        return y_pred


class StackBoostRegressor(StackBoost):
    def __init__(self, n_estimators=100, learning_rate=0.001, min_samples_split=2,
                 min_impurity=1e-7, max_depth=2, loss_function=SquareLoss(), l2_leaf_reg: float = 0.0,
                 n_quantiles: int = 33,
                 max_leaf_samples: int = 1000, prune: bool = False, gamma_percentage: float = 1., metrics: list = [],
                 verbose: bool = False, speedup: bool = False):
        super(StackBoostRegressor, self).__init__(n_estimators=n_estimators,
                                           learning_rate=learning_rate,
                                           min_samples_split=min_samples_split,
                                           min_impurity=min_impurity,
                                           max_depth=max_depth,
                                           loss_function=loss_function,
                                           l2_leaf_reg=l2_leaf_reg,
                                           n_quantiles=n_quantiles,
                                           max_leaf_samples=max_leaf_samples,
                                           prune=prune,
                                           gamma_percentage=gamma_percentage,
                                           metrics=metrics,
                                           verbose=verbose,
                                           regression=True,
                                           speedup=speedup)

    def fit(self, X, y, cat_features=None, sample_weight=None, eval_set=None, use_best_model=False):
        y = y.reshape(-1, 1)
        super(StackBoostRegressor, self).fit(X, y, cat_features, sample_weight, eval_set, use_best_model)


class StackBoostClassifier(StackBoost):
    def __init__(self, n_estimators=100, learning_rate=0.001, min_samples_split=2,
                 min_impurity=1e-7, max_depth=2, loss_function=LogisticLoss(), l2_leaf_reg: float = 0.0,
                 n_quantiles: int = 33, max_leaf_samples: int = 1000, prune: bool = False,
                 gamma_percentage: float = 1., metrics: list = [], verbose: bool = False, speedup: bool = False,
                 activation=Sigmoid()):
        super(StackBoostClassifier, self).__init__(n_estimators=n_estimators,
                                            learning_rate=learning_rate,
                                            min_samples_split=min_samples_split,
                                            min_impurity=min_impurity,
                                            max_depth=max_depth,
                                            loss_function=loss_function,
                                            l2_leaf_reg=l2_leaf_reg,
                                            n_quantiles=n_quantiles,
                                            max_leaf_samples=max_leaf_samples,
                                            prune=prune,
                                            gamma_percentage=gamma_percentage,
                                            metrics=metrics,
                                            verbose=verbose,
                                            regression=False,
                                            speedup=speedup)
        self.activation = activation

    def fit(self, X, y, cat_features=None, sample_weight=None, eval_set=None, use_best_model=False):
        y = to_categorical(y)
        super(StackBoostClassifier, self).fit(X, y, cat_features, sample_weight, eval_set, use_best_model)

    def predict_proba(self, X):
        y_pred = super(StackBoostClassifier, self).predict(X)
        y_pred = self.activation(y_pred)
        return y_pred

    def predict(self, X):
        y_pred = self.predict_proba(X)
        y_pred = np.argmax(y_pred, axis=1)
        return y_pred
