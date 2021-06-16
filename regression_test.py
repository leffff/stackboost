from stackboost.core import StackBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.datasets import load_boston
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from stackboost.utils.loss_functions import SquareLoss
from catboost import CatBoostRegressor

rocs = []
accus = []
mses = []

X, y = load_boston(return_X_y=True)
kfold = KFold(n_splits=4, shuffle=True, random_state=15)
for train_index, test_index in kfold.split(X, y):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    xgb = StackBoostRegressor(
        sketch_type="minmax",
        verbose=True,
        loss_function=SquareLoss(),
        n_estimators=50,
        n_quintiles=80,
        metrics=[mean_squared_error],
        max_depth=4,
        l2_leaf_reg=2,
        speedup=True,
        undersampling_percentage=0.1
    )
    xgb.fit(X_train, y_train)
    preds = xgb.predict(X_test)

    mse = mean_squared_error(y_test, preds)
    print(mse)
    mses.append(mse)

print(sum(mses) / len(mses))
