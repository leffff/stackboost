from __future__ import division, print_function
import numpy as np

from stackboost.utils.data_manipulation import divide_on_feature, to_array
from stackboost.utils.data_operation import calculate_entropy, mean_squared_error, similarity_score, calculate_variance
from stackboost.utils.categorical_encoding import StackedEncoder
import torch
import pandas as pd
import warnings

warnings.filterwarnings("ignore")
SUPPORTED_ARRAYS = [np.ndarray, torch.tensor, pd.Series, pd.DataFrame, list]


class DecisionNode:
    def __init__(self, feature_i=None, threshold=None, value=None, target=None, true_branch=None,
                 false_branch=None):
        self.feature_i = feature_i  # Index for the feature that is tested
        self.threshold = threshold  # Threshold value for feature
        self.value = value  # Value if the node is a leaf in the tree
        self.target = target
        self.true_branch = true_branch  # 'Left' subtree
        self.false_branch = false_branch  # 'Right' subtree

    def is_leaf(self):
        return self.value != None


class DecisionTree(object):
    def __init__(self, min_samples_split: int = 2, min_impurity: float = 1e-7,
                 max_depth: int = float("inf"), l2_leaf_reg: float = 0.0, n_quantiles: int = 33,
                 max_leaf_samples: int = 1000, prune: bool = False, gamma: int = 0, loss_function=mean_squared_error):

        self.root = None  # Root node in dec. tree
        # Minimum n of samples to justify split
        self.min_samples_split = min_samples_split
        # The minimum impurity to justify split
        self.min_impurity = min_impurity
        # The maximum depth to grow the tree to
        self.max_depth = max_depth
        self.l2_leaf_reg = l2_leaf_reg
        self.n_quantiles = n_quantiles
        self.max_leaf_samples = max_leaf_samples
        # Function to calculate impurity (classif.=>info gain, regr=>variance reduct.)
        self._impurity_calculation = None
        # Function to determine prediction of y at leaf
        self.leaf_training = None
        # If y is one-hot encoded (multi-dim) or not (one-dim)
        self.one_dim = None
        # If Gradient Boost
        self.cat_features = None
        self.sample_weight = None
        self.prune = prune
        self.gamma = gamma
        self.loss_function = loss_function
        self.encoder = StackedEncoder()
        self.n_leaves = 0
        self.used_features = []

    def fit(self, X, y, sample_weight=None, cat_features=None):
        """ Build decision tree """
        X, y = to_array(X), to_array(y)
        self.cat_features = cat_features
        if self.cat_features in SUPPORTED_ARRAYS:
            X = self.encoder.fit_transform(X, y, cat_features=self.cat_features)

        self.sample_weight = sample_weight
        self.n_features = X.shape[1]
        if type(sample_weight) not in SUPPORTED_ARRAYS:
            self.sample_weight = 1 / np.array([1 for i in range(self.n_features)])

        self.n_leaves = 0
        self.one_dim = len(np.shape(y)) == 1
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

        Xy = np.concatenate((X, y), axis=1)

        n_samples, n_features = np.shape(X)

        if n_samples >= self.min_samples_split and current_depth <= self.max_depth and self.n_leaves < self.max_leaf_samples:
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

                # Iterate through all unique values of feature column i and
                # calculate the impurity
                for threshold in quintiles:
                    # Divide X and y depending on if the feature value of X at index feature_i
                    # meets the threshold
                    Xy1, Xy2 = divide_on_feature(Xy, feature_i, threshold)

                    if len(Xy1) > 0 and len(Xy2) > 0:
                        # Select the y-values of the two sets

                        # Calculate impurity

                        impurity = self._impurity_calculation(y, Xy1, Xy2) * self.sample_weight[feature_i]

                        # If this threshold resulted in a higher information gain than previously
                        # recorded save the threshold value and the feature
                        # index
                        if impurity > largest_impurity:
                            largest_impurity = impurity
                            best_criteria = {"feature_i": feature_i, "threshold": threshold}
                            best_sets = {
                                "leftX": Xy1[:, :n_features],  # X of left subtree
                                "lefty": Xy1[:, n_features:],  # y of left subtree
                                "rightX": Xy2[:, :n_features],  # X of right subtree
                                "righty": Xy2[:, n_features:]  # y of right subtree
                            }

                            self.used_features.append(feature_i)

        if largest_impurity > self.min_impurity:
            # Build subtrees for the right and left branches
            true_branch = self._build_tree(best_sets["leftX"], best_sets["lefty"], current_depth + 1)
            false_branch = self._build_tree(best_sets["rightX"], best_sets["righty"], current_depth + 1)

            new_node = DecisionNode(feature_i=best_criteria["feature_i"], threshold=best_criteria[
                "threshold"], true_branch=true_branch, false_branch=false_branch, target=y)
            return new_node

        # We're at leaf => determine value
        leaf_value = self.leaf_training(y)
        self.n_leaves += 1
        return DecisionNode(value=leaf_value, target=y)

    def predict_value(self, X, tree=None):
        """ Do a recursive search down the tree and make a prediction of the data sample by the
            value of the leaf that we end up at """

        if tree is None:
            tree = self.root

        # If we have a value (i.e we're at a leaf) => return value as the prediction
        if tree.value is not None:
            return tree.value

        # Choose the feature that we will test
        feature_value = X[tree.feature_i]

        # Determine if we will follow left or right branch
        branch = tree.false_branch
        if isinstance(feature_value, int) or isinstance(feature_value, float):
            if feature_value >= tree.threshold:
                branch = tree.true_branch
        elif feature_value == tree.threshold:
            branch = tree.true_branch

        # Test subtree
        return self.predict_value(X, branch)

    def predict(self, X):
        """ Classify samples one by one and return the set of labels """
        if self.cat_features in SUPPORTED_ARRAYS:
            print("yes")
            X = self.encoder.transform(X)

        y_pred = np.array([self.predict_value(sample) for sample in X])
        return y_pred

    def get_feature_importance(self):
        importance_count = {}
        for feature in range(self.n_features):
            importance_count[feature] = 0

        for features in self.used_features:
            importance_count[features] += 1

        all_counts = sum(list(importance_count.values()))

        perc_importance = {}
        for feature in range(self.n_features):
            perc_importance[feature] = importance_count.get(feature) / all_counts

        return perc_importance

    def __prune(self, tree=None):
        if tree is None:
            tree = self.root

        # if node is not a leaf
        if tree.true_branch != None:
            tree.true_branch = self.__prune(tree.true_branch)
        if tree.false_branch != None:
            tree.false_branch = self.__prune(tree.false_branch)

        print("------------------")
        if not tree.is_leaf():
            print(tree.true_branch.is_leaf(), tree.false_branch.is_leaf())

        if not tree.is_leaf() and tree.true_branch.is_leaf() and tree.false_branch.is_leaf():
            print("XXXXXXXXXXXXXXXXX")
            if similarity_score(np.append(tree.false_branch.target, tree.true_branch.target),
                                self.l2_leaf_reg) - self.gamma < 0:
                return DecisionNode(value=self.leaf_training(tree.target), target=tree.target)

        return tree


class SimilarityTreeRegressor(DecisionTree):
    def _calculate_variance_reduction(self, y, Xy1, Xy2):
        y1, X1 = Xy1[:, self.n_features:], Xy1[:, :self.n_features]
        y2, X2 = Xy2[:, self.n_features:], Xy2[:, :self.n_features]

        gain_1 = similarity_score(y1, self.l2_leaf_reg)
        gain_2 = similarity_score(y2, self.l2_leaf_reg)
        gain_tot = similarity_score(y, self.l2_leaf_reg)

        variance_reduction = gain_1 + gain_2 - gain_tot
        return variance_reduction

    def regression_leaf_training(self, y):
        value = np.mean(y, axis=0)
        return value if len(value) > 1 else value[0]

    def fit(self, X, y, sample_weight=None, cat_features=None):
        self._impurity_calculation = self._calculate_variance_reduction
        self.leaf_training = self.regression_leaf_training
        super(SimilarityTreeRegressor, self).fit(X, y, sample_weight, cat_features)


class TreeRegressor(DecisionTree):
    def _calculate_variance_reduction(self, y, y1, y2):
        var_tot = calculate_variance(y)
        var_1 = calculate_variance(y1)
        var_2 = calculate_variance(y2)
        frac_1 = len(y1) / len(y)
        frac_2 = len(y2) / len(y)

        # Calculate the variance reduction
        variance_reduction = var_tot - (frac_1 * var_1 + frac_2 * var_2)
        return variance_reduction

    def _mean_of_y(self, y):
        value = np.mean(y, axis=0)
        return value if len(value) > 1 else value[0]

    def fit(self, X, y, sample_weight=None, cat_features=None):
        self._impurity_calculation = self._calculate_variance_reduction
        self.leaf_training = self._mean_of_y
        super(TreeRegressor, self).fit(X, y, sample_weight, cat_features)



class TreeClassifier(DecisionTree):
    def _calculate_information_gain(self, y, Xy1, Xy2):
        y1 = Xy1[:, self.n_features:]
        y2 = Xy2[:, self.n_features:]
        # Calculate information gain
        p = len(y1) / len(y)
        entropy = calculate_entropy(y)
        info_gain = entropy - p * calculate_entropy(y1) - (1 - p) * calculate_entropy(y2)

        return info_gain

    def classification_leaf_training(self, y):
        print(y)
        most_common = None
        max_count = 0
        for label in np.unique(y):
            # Count number of occurences of samples with label
            count = len(y[y == label])
            if count > max_count:
                most_common = label
                max_count = count
        return most_common

    def fit(self, X, y, sample_weight=None, cat_features=None):
        self._impurity_calculation = self._calculate_information_gain
        self.leaf_training = self.classification_leaf_training
        super(TreeClassifier, self).fit(X, y, sample_weight, cat_features)

    def predict(self, X):
        pass

    def predict_proba(self, X):
        pass