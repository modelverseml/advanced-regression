# Regularization Example: Lasso, Ridge, Elastic Net

import numpy as np
from sklearn.linear_model import Lasso, Ridge, ElasticNet
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# 1. Generate synthetic regression data
X, y = make_regression(
    n_samples=100, 
    n_features=10, 
    noise=10, 
    random_state=42
)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


alpha = 0.5 

lasso = Lasso(alpha=alpha)
ridge = Ridge(alpha=alpha)
elastic_net = ElasticNet(alpha=alpha, l1_ratio=0.5)


lasso.fit(X_train, y_train)
ridge.fit(X_train, y_train)
elastic_net.fit(X_train, y_train)


y_pred_lasso = lasso.predict(X_test)
y_pred_ridge = ridge.predict(X_test)
y_pred_en = elastic_net.predict(X_test)


print("Mean Squared Error:")
print("Lasso:      ", mean_squared_error(y_test, y_pred_lasso))
print("Ridge:      ", mean_squared_error(y_test, y_pred_ridge))
print("ElasticNet: ", mean_squared_error(y_test, y_pred_en))

