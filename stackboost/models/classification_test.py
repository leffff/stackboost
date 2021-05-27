from stackboost.models.sgboost import SGBClassifier, SGBRegressor
from sklearn.tree import DecisionTreeClassifier
# from sklearn.ensemble import GradientBoostingClassifier
from stackboost.models.gradient_boosting import GradientBoostingClassifier
from sklearn.datasets import load_breast_cancer, load_boston, load_iris
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score, accuracy_score, mean_squared_error
rocs = []
accus = []
mses = []

X, y = load_iris(return_X_y=True)
kfold = KFold(n_splits=3, shuffle=True, random_state=42)
for train_index, test_index in kfold.split(X, y):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    # print(len(train_index), len(test_index))
    xgb = SGBClassifier(verbose=True, n_estimators=100, n_quantiles=40, metrics=[accuracy_score])
    xgb.fit(X_train, y_train, eval_set=(X_test, y_test))
    preds = xgb.predict(X_test)
    print(xgb.predict_proba(X_test))
    # mse = mean_squared_error(y_test, preds)
    # print(mse)
    # roc_auc = roc_auc_score(y_test, preds)
    accuracy = accuracy_score(y_test, preds)
    # print(roc_auc, accuracy)
    # rocs.append(roc_auc)
    accus.append(accuracy)

# print(sum(mses) / len(mses))
# print(sum(rocs) / len(rocs))
print(sum(accus) / len(accus))

