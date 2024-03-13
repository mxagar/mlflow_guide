'''Housing price prediction project.
MLflow is used for tracking and the project is deployed on AWS.

This module performs the model training.

Original code from the Udemy course

    MLflow in Action - Master the art of MLOps using MLflow tool
    https://www.udemy.com/course/mlflow-course/

    Author: J Garg, Real Time Learning

'''
import mlflow
import numpy as np
from sklearn.linear_model import Ridge, ElasticNet
from xgboost import XGBRegressor
from sklearn.model_selection import ParameterGrid
from data import X_train, X_val, y_train, y_val
from params import ridge_param_grid, elasticnet_param_grid, xgb_param_grid
from eval import eval_metrics

# Model: ElasticNet
# NOTE: We usually don't hard-code any experiment/run name/ids,
# these are set dynamically
# and here MLproject contains the important configuration,
# so that the script/code is generic!
# Also, it is common to have a run.py file which uses hard-coded values (i.e., URIs).
# Loop through the hyperparameter combinations and log results in separate runs
for params in ParameterGrid(elasticnet_param_grid):
    with mlflow.start_run():
        # Fir model
        lr = ElasticNet(**params)
        lr.fit(X_train, y_train)

        # Evaluate trained model
        y_pred = lr.predict(X_val)
        metrics = eval_metrics(y_val, y_pred)

        # Logging the inputs such as dataset
        mlflow.log_input(
            mlflow.data.from_numpy(X_train.toarray()),
            context='Training dataset'
        )
        mlflow.log_input(
            mlflow.data.from_numpy(X_val.toarray()),
            context='Validation dataset'
        )

        # Log hyperparameters
        mlflow.log_params(params)

        # Log metrics
        mlflow.log_metrics(metrics)

        # Log the trained model
        mlflow.sklearn.log_model(
            lr,
            "ElasticNet",
             input_example=X_train,
             # Log the files used for training, too!
             code_paths=['train.py','data.py','params.py','eval.py']
        )
