from __future__ import division, print_function
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from stackboost.utils.data_manipulation import divide_on_feature, to_array, to_categorical
from stackboost.utils.optimization import min_max_sketch, hist_sketch, greedy_sketch, cluster_sketch
from stackboost.utils.optimization import make_bins, undersampling
from stackboost.utils.loss_functions import LogisticLoss, SquareLoss
from stackboost.utils.activation_functions import Sigmoid, Softmax
from stackboost.utils.errors import NotFittedError

import warnings

warnings.filterwarnings("ignore")


class DecisionNode:
    """
    DecisionNode is a basis unit of a decision tree. Decision tree are made of DecisionNodes linked together
    """

    def __init__(self, feature_i: int = None, threshold: float = None, value: np.ndarray = None,
                 target: np.ndarray = None, true_branch=None, false_branch=None, importance: float = None,
                 depth: int = None):
        """
        :param feature_i:
            The feature index the data is split on.
        :param threshold:
            The threshold feature is devided on.
        :param value:
            The value of the node if the node is a leaf.
        :param target:
            Target va.ues of the leaf.
        :param true_branch:
            The decision node data goes to if it is greater than the threshold.
        :param false_branch:
            The decision node data goes to if it is smaller than the threshold.
        :param importance:
            The importance of the node in the tree used to calculate feature importances.
        :param depth:
            The depth of the current node in the tree.
        """

        self.feature_i = feature_i
        self.threshold = threshold
        self.value = value
        self.target = target
        self.true_branch = true_branch
        self.false_branch = false_branch
        self.importance = importance
        self.depth = depth

    def is_leaf(self):
        """
        The function used to check whether the DecisionNode is a leaf.
        """
        return not self.value is None


class DepthwiseTree(object):
    """
    Super class of StackBoostTree.
    """

    def __init__(self, loss_function, sketch_algorithm, min_samples_split: int = 2, min_impurity: float = 1e-7,
                 max_depth: int = float("inf"), l2_leaf_reg: float = 0.0, n_quintiles: int = 33,
                 prune: bool = False, gamma_percentage: float = 1.0, position=None, speedup: bool = False):
        """
        :param loss_function
            The loss function used to optimize the leaf values.
        :param min_samples_split: int
            The minimum number of samples needed to make a split when building a tree.
        :param min_impurity: float
            The minimum impurity required to split the tree further.
        :param max_depth: int
            The maximum depth of a tree.
        :param l2_leaf_reg: float
            The regularization parameter also known as lambda (λ). used for shrinking output values and encourage pruning.
        :param n_quintiles: int
            The number of thresholds we use to choose fom on each iterration.
        :param prune: bool
            The flag that defines whether we prune the Decision tree after it is built.
        :param gamma_percentage: float
            The parameter that encourages pruning (γ).
        :param position: int
            The position of an estimator in the set of estimators.
        :param speedup: bool
            The flag that enables gradient speedup. Multiply gradient by position of the tree in the sequence if True.
        """

        # init params
        self.loss_function = loss_function
        self.sketch_algorithm = sketch_algorithm
        self.min_samples_split = min_samples_split
        self.min_impurity = min_impurity
        self.max_depth = max_depth
        self.l2_leaf_reg = l2_leaf_reg
        self.n_quintiles = n_quintiles
        self.prune = prune
        self.gamma_percentage = gamma_percentage
        self.position = position
        self.speedup = speedup

        self.root = None
        self._impurity_calculation = None
        self._leaf_value_calculation = None
        self.one_dim = None
        self.fitted = False
        self.current_depth = 0
        self.current_leaf_samples = 0
        self.n_nodes = 1
        self.total_gain = 0
        self.average_gain = 0

        self.sample_weight = None

    def fit(self, X: np.ndarray, y: np.ndarray, sample_weight: np.ndarray) -> None:
        """
        Main training function

        :param X: training data
        :param y: target data
        :param sample_weight: weights for _impurity_calculation
        :return: None
        """
        self.one_dim = len(np.shape(y)) == 1
        self.n_samples, self.n_features = X.shape

        self.sample_weight = sample_weight

        self.root = self._build_tree(X, y)
        self.average_gain = self.total_gain / self.n_nodes

        if self.prune:
            self.root = self.__prune()

        self.fitted = True

    def _build_tree(self, X: np.ndarray, y: np.ndarray, current_depth: int = 0) -> DecisionNode:
        """
        Recursive method which builds the decision tree and splits X and respective y
        on the feature of X which best separates the data

        :param X: training data
        :param y: target data
        :param current_depth: current depth of built tree
        :return:
        """

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

                quintiles = self.sketch_algorithm(X=feature_values, n_quantiles=self.n_quintiles)

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

    def __prune(self, tree: DecisionNode = None) -> DecisionNode:
        """
        Method used to prune the decision tree in order to avoid overfitting

        :param tree: Sub tree of the decision tree (node)
        :return: DecisionNode
        """
        if tree is None:
            tree = self.root

        if not tree.is_leaf() and not tree.true_branch.is_leaf() and not tree.false_branch.is_leaf():
            tree.true_branch = self.__prune(tree.true_branch)
            tree.false_branch = self.__prune(tree.false_branch)

        if not tree.is_leaf() and tree.true_branch.is_leaf() and tree.false_branch.is_leaf():
            if self._impurity_calculation(tree.target, tree.true_branch.target,
                                          tree.false_branch.target) - self.average_gain * self.gamma_percentage < 0:
                return DecisionNode(value=self._leaf_value_calculation(tree.target), target=tree.target)

        return tree

    def __calculate_importance(self, importance: np.ndarray, tree: DecisionNode = None) -> np.ndarray:
        """
        Inner function used to calculated feature importances

        :param importance: array of importances
        :param tree: the current tree node
        :return: array of importances
        """
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
        """
        Method used to calculate feature importances

        :return: array of feature importances
        """
        if not self.fitted:
            raise NotFittedError()

        feature_importance = self.__calculate_importance(self.sample_weight)
        return feature_importance

    def predict_value(self, X: np.ndarray, tree: DecisionNode = None) -> np.ndarray:
        """ Do a recursive search down the tree and make a prediction of the data sample by the
            value of the leaf that we end up at """

        if tree is None:
            tree = self.root

        # If we have a value (i.e we're at a leaf) => return value as the prediction
        if tree.is_leaf():
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

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.fitted:
            raise NotFittedError()

        y_pred = np.vstack([self.predict_value(sample) for sample in X])
        return y_pred


class StackBoostTree(DepthwiseTree):
    def _split(self, y: np.ndarray) -> [np.ndarray, np.ndarray]:
        col = int(np.shape(y)[1] / 2)
        y, y_pred = y[:, :col], y[:, col:]
        return y, y_pred

    def _gain(self, y: np.ndarray, y_pred: np.ndarray) -> float:
        nominator = np.power((y * self.loss_function.gradient(y, y_pred)).sum(), 2)
        denominator = self.loss_function.hess(y, y_pred).sum()
        return 0.5 * (nominator / denominator)

    def _gain_by_taylor(self, y: np.ndarray, y1: np.ndarray, y2: np.ndarray) -> float:
        # Split
        y, y_pred = self._split(y)
        y1, y1_pred = self._split(y1)
        y2, y2_pred = self._split(y2)
        # calculate gain
        true_gain = self._gain(y1, y1_pred)
        false_gain = self._gain(y2, y2_pred)
        gain = self._gain(y, y_pred)
        return true_gain + false_gain - gain

    def _approximate_update(self, y: np.ndarray) -> float:
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

    def fit(self, X: np.ndarray, y: np.ndarray, sample_weight: np.ndarray) -> None:
        self._impurity_calculation = self._gain_by_taylor
        self._leaf_value_calculation = self._approximate_update
        super(StackBoostTree, self).fit(X, y, sample_weight)


class StackBoost(object):
    """
    Super class of StackBoostRegressor and StackBoostClassifier.
    """

    def __init__(self, n_estimators: int = 100, learning_rate: float = 0.001, loss_function=LogisticLoss(),
                 min_samples_split: int = 2, min_impurity: float = 1e-7, max_depth: int = 2, l2_leaf_reg: float = 0.0,
                 n_quintiles: int = 33, prune: bool = False, gamma_percentage: float = 1.0,
                 undersampling_percentage: float = 0.0, sketch_type: str = "minmax", speedup: bool = False,
                 metrics: list = None, verbose: bool = False, regression=False):
        """
        :param n_estimators:
            The number of estimators
        :param learning rate:
            The number by which we scale each tree in the sequence. Also known as eta (η).
        :param loss_function:
            The loss function used to optimize the leaf values.
        :param min_samples_split:
            The minimum number of samples needed to make a split when building a tree.
        :param min_impurity:
            The minimum impurity required to split the tree further.
        :param max_depth:
            The maximum depth of a tree.
        :param l2_leaf_reg:
            The regularization parameter also known as lambda (λ). used for shrinking output values and encourage pruning.
        :param n_quintiles:
            The number of thresholds we use to choose fom on each iterration.
        :param prune:
            The flag that defines whether we prune the Decision tree after it is built.
        :param gamma_percentage:
            The parameter that encourages pruning (γ).
        :param undersampling_percentage:
            The percent of samples with smallest gradients to delet on each iterration
        :param sketch_type:
            Type of algorithm for figuring out the best thresholds
        :param speedup:
            The flag that enables gradient speedup. Multiply gradient by position of the tree in the sequence if True.
        :param mertics:
            The array of metric functions.
        :param verbose:
            The flag, which enables training information.
        :param regression:
            The flag, which defines a task type: regression if True classification in ohter case
        """

        self.n_estimators = n_estimators  # Number of trees
        self.learning_rate = learning_rate  # Step size for weight update
        self.min_samples_split = min_samples_split  # The minimum n of sampels to justify split
        self.min_impurity = min_impurity  # Minimum variance reduction to continue
        self.max_depth = max_depth
        self.l2_leaf_reg = l2_leaf_reg
        self.n_quintiles = n_quintiles
        self.prune = prune
        self.gamma_percentage = gamma_percentage
        self.undersampling_percentage = undersampling_percentage
        self.sketch_type = sketch_type
        sketch_types = {"minmax": min_max_sketch, "hist": hist_sketch, "greedy": greedy_sketch}
        self.sketch_algorithm = sketch_types.get(self.sketch_type)
        self.speedup = speedup
        self.metrics = metrics
        if metrics is None:
            self.metrics = []
        self.verbose = verbose
        self.regression = regression

        self.fitted = False

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
                sketch_algorithm=self.sketch_algorithm,
                loss_function=self.loss_function,
                min_samples_split=self.min_samples_split,
                min_impurity=self.min_impurity,
                max_depth=self.max_depth,
                l2_leaf_reg=self.l2_leaf_reg,
                n_quintiles=self.n_quintiles,
                prune=self.prune,
                gamma_percentage=self.gamma_percentage,
                position=_,
                speedup=self.speedup
            )

            self.trees.append(tree)

    def fit(self, X: np.ndarray, y: np.ndarray, cat_features: [list, np.ndarray], sample_weight: np.ndarray,
            eval_set: [list, tuple], use_best_model: bool) -> None:
        """
        Main training method

        :param X: training dataset
        :param y: train targets
        :param cat_features: categorical features
        :param sample_weight: feature weitghts
        :param eval_set: validation set
        :param use_best_model: flag that enables usage of the first n best trees
        :return: None
        """
        X, y = to_array(X), to_array(y)

        self.n_samples, self.n_features = X.shape

        self.sample_weight = sample_weight
        if self.sample_weight is None:
            self.sample_weight = 1 / np.array([1 for i in range(self.n_features)])

        self.cat_features = cat_features
        if cat_features is None:
            self.cat_features = []

        if cat_features is not None:
            X_test, y_test = eval_set

        self.use_best_model = use_best_model

        self.initial_prediction = y.mean() if self.regression else float(np.argmax(np.mean(y, axis=0)))
        y_pred = np.full(np.shape(y), self.initial_prediction)

        self.test_errors = []

        sample_mask = np.array([True for i in range(self.n_samples)])

        for i in range(self.n_estimators):
            if self.undersampling_percentage != 0:
                grads = self.loss_function.gradient(y, y_pred)
                sample_mask = undersampling(grads, self.undersampling_percentage)

            y_and_pred = np.concatenate((y, y_pred), axis=1)
            sub_X, sub_y_and_pred = X[sample_mask], y_and_pred[sample_mask]

            tree = self.trees[i]
            tree.fit(sub_X, sub_y_and_pred, sample_weight=self.sample_weight)
            preds = tree.predict(X)
            update_pred = np.multiply(self.learning_rate, preds)

            y_pred -= update_pred

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
                    if eval_set is not None:
                        if self.regression:
                            test_scores = [scoring_function(y_test, self.__inner_predict(X_test, i)) for
                                           scoring_function in self.metrics]

                            self.test_errors.append(
                                self.loss_function.loss(y_test, self.__inner_predict(X_test, i)))

                        else:
                            test_scores = [
                                scoring_function(y_test, np.argmax(self.__inner_predict(X_test, i), axis=1))
                                for scoring_function in self.metrics]

                            self.test_errors.append(self.loss_function.loss(y_test, np.argmax(
                                self.__inner_predict(X_test, i), axis=1)))

                        test_scoring_output = f"test metrics: {test_scores}"

                print(f"Tree: {i}", train_scoring_output, test_scoring_output)

        if self.use_best_model:
            self.best_iteration = np.argmax(self.test_errors)
        else:
            self.best_iteration = self.n_estimators

        self.fitted = True

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Main prediction method

        :param X: prediction dataset
        :return: predictions array
        """
        if not self.fitted:
            raise NotFittedError()

        return self.__inner_predict(X, self.best_iteration)

    def __inner_predict(self, X: np.ndarray, iteration: int) -> np.ndarray:
        """
        Inner method used to make prediction of n trees

        :param X: prediction dataset
        :param iteration: number of trees to predict
        :return: predictions array
        """
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
    """
    Super class of StackBoostRegressor and StackBoostClassifier.
    """

    def __init__(self, n_estimators: int = 100, learning_rate: float = 0.001, loss_function=SquareLoss(),
                 min_samples_split: int = 2, min_impurity: float = 1e-7, max_depth: int = 2, l2_leaf_reg: float = 0.0,
                 n_quintiles: int = 33, prune: bool = False, gamma_percentage: float = 1.,
                 undersampling_percentage: float = 0.0, sketch_type: str = "minmax", speedup: bool = False,
                 metrics: list = None, verbose: bool = False):
        """
        :param n_estimators:
            The number of estimators
        :param learning rate:
            The number by which we scale each tree in the sequence. Also known as eta (η).
        :param loss_function:
            The loss function used to optimize the leaf values.
        :param min_samples_split:
            The minimum number of samples needed to make a split when building a tree.
        :param min_impurity:
            The minimum impurity required to split the tree further.
        :param max_depth:
            The maximum depth of a tree.
        :param l2_leaf_reg:
            The regularization parameter also known as lambda (λ). used for shrinking output values and encourage pruning.
        :param n_quintiles:
            The number of thresholds we use to choose fom on each iterration.
        :param prune:
            The flag that defines whether we prune the Decision tree after it is built.
        :param gamma_percentage:
            The parameter that encourages pruning (γ).
        :param undersampling_percentage:
            The percent of samples with smallest gradients to delet on each iterration
        :param sketch_type:
            Type of algorithm for figuring out the best thresholds
        :param speedup:
            The flag that enables gradient speedup. Multiply gradient by position of the tree in the sequence if True.
        :param mertics:
            The array of metric functions.
        :param verbose:
            The flag, which enables training information.
        """
        super(StackBoostRegressor, self).__init__(n_estimators=n_estimators,
                                                  learning_rate=learning_rate,
                                                  min_samples_split=min_samples_split,
                                                  min_impurity=min_impurity,
                                                  max_depth=max_depth,
                                                  loss_function=loss_function,
                                                  l2_leaf_reg=l2_leaf_reg,
                                                  n_quintiles=n_quintiles,
                                                  prune=prune,
                                                  gamma_percentage=gamma_percentage,
                                                  undersampling_percentage=undersampling_percentage,
                                                  sketch_type=sketch_type,
                                                  metrics=metrics,
                                                  verbose=verbose,
                                                  regression=True,
                                                  speedup=speedup)

    def fit(self, X, y, cat_features: [list, tuple, np.ndarray] = None, sample_weight: np.ndarray = None,
            eval_set: [list, tuple] = None, use_best_model: bool = False) -> None:
        """
        Main training method

        :param X: training dataset
        :param y: train targets
        :param cat_features: categorical features
        :param sample_weight: feature weitghts
        :param eval_set: validation set
        :param use_best_model: flag that enables usage of the first n best trees
        :return: None
        """
        y = y.reshape(-1, 1)
        super(StackBoostRegressor, self).fit(X, y, cat_features, sample_weight, eval_set, use_best_model)


class StackBoostClassifier(StackBoost):
    """
    Super class of StackBoostRegressor and StackBoostClassifier.
    """

    def __init__(self, n_estimators: int = 100, learning_rate: float = 0.001, min_samples_split: int = 2,
                 min_impurity: float = 1e-7, max_depth: int = 2, loss_function=LogisticLoss(), l2_leaf_reg: float = 0.0,
                 n_quintiles: int = 33, prune: bool = False, gamma_percentage: float = 1.,
                 undersampling_percentage: float = 0.0, sketch_type: str = "minmax",
                 speedup: bool = False, metrics: list = None, verbose: bool = False, activation=Sigmoid()):
        """
        :param n_estimators:
            The number of estimators
        :param learning rate:
            The number by which we scale each tree in the sequence. Also known as eta (η).
        :param loss_function:
            The loss function used to optimize the leaf values.
        :param min_samples_split:
            The minimum number of samples needed to make a split when building a tree.
        :param min_impurity:
            The minimum impurity required to split the tree further.
        :param max_depth:
            The maximum depth of a tree.
        :param l2_leaf_reg:
            The regularization parameter also known as lambda (λ). used for shrinking output values and encourage pruning.
        :param n_quintiles:
            The number of thresholds we use to choose fom on each iterration.
        :param prune:
            The flag that defines whether we prune the Decision tree after it is built.
        :param gamma_percentage:
            The parameter that encourages pruning (γ).
        :param undersampling_percentage:
            The percent of samples with smallest gradients to delet on each iterration
        :param sketch_type:
            Type of algorithm for figuring out the best thresholds
        :param speedup:
            The flag that enables gradient speedup. Multiply gradient by position of the tree in the sequence if True.
        :param mertics:
            The array of metric functions.
        :param verbose:
            The flag, which enables training information.
        :param activation:
            The activation function used to calculate probabilities.
        """
        super(StackBoostClassifier, self).__init__(n_estimators=n_estimators,
                                                   learning_rate=learning_rate,
                                                   min_samples_split=min_samples_split,
                                                   min_impurity=min_impurity,
                                                   max_depth=max_depth,
                                                   loss_function=loss_function,
                                                   l2_leaf_reg=l2_leaf_reg,
                                                   n_quintiles=n_quintiles,
                                                   prune=prune,
                                                   gamma_percentage=gamma_percentage,
                                                   undersampling_percentage=undersampling_percentage,
                                                   sketch_type=sketch_type,
                                                   metrics=metrics,
                                                   verbose=verbose,
                                                   regression=False,
                                                   speedup=speedup)
        self.activation = activation

    def fit(self, X, y, cat_features: [list, tuple, np.ndarray] = None, sample_weight: np.ndarray = None,
            eval_set: [list, tuple] = None, use_best_model: bool = False) -> None:
        """
        Main training method

        :param X: training dataset
        :param y: train targets
        :param cat_features: categorical features
        :param sample_weight: feature weitghts
        :param eval_set: validation set
        :param use_best_model: flag that enables usage of the first n best trees
        :return: None
        """
        y = to_categorical(y)
        super(StackBoostClassifier, self).fit(X, y, cat_features, sample_weight, eval_set, use_best_model)

    def predict_proba(self, X) -> np.ndarray:
        """
        Method for probability prediction

        :param X: prediction dataset
        :return: prediction of class probabilities
        """
        if not self.fitted:
            raise NotFittedError()

        y_pred = super(StackBoostClassifier, self).predict(X)
        y_pred = self.activation(y_pred)
        return y_pred

    def predict(self, X) -> np.ndarray:
        """
        Method for class prediction

        :param X: prediction dataset
        :return: prediction of classes
        """
        if not self.fitted:
            raise NotFittedError()

        y_pred = self.predict_proba(X)
        y_pred = np.argmax(y_pred, axis=1)
        return y_pred
