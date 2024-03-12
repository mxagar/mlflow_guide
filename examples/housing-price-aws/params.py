
# Ridge Param Grid
ridge_param_grid = {
    'alpha': [0.1, 1.0, 10.0],
    'fit_intercept': [True, False],
}

elasticnet_param_grid = {
    'alpha': [0.1, 1.0, 10.0],
    'l1_ratio': [0.2, 0.5, 0.8],
    'fit_intercept': [True, False],
}


xgb_param_grid = {
    'n_estimators': [100, 200, 300],  # Number of boosting rounds
    'learning_rate': [0.01, 0.1, 0.2],  # Step size shrinkage to prevent overfitting
    'max_depth': [3, 4, 5],  # Maximum depth of each tree
    'min_child_weight': [1, 2, 3],  # Minimum sum of instance weight needed in a child
    'subsample': [0.8, 0.9, 1.0],  # Fraction of samples used for training
    'colsample_bytree': [0.8, 0.9, 1.0],  # Fraction of features used for training
    'gamma': [0, 0.1, 0.2],  # Minimum loss reduction required to make a further partition on a leaf node
    'reg_alpha': [0, 0.1, 1.0],  # L1 regularization term on weights
    'reg_lambda': [0, 0.1, 1.0],  # L2 regularization term on weights
}
