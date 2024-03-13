'''Housing price prediction project.
MLflow is used for tracking and the project is deployed on AWS.

This module performs the model training.

Original code from the Udemy course

    MLflow in Action - Master the art of MLOps using MLflow tool
    https://www.udemy.com/course/mlflow-course/

    Author: J Garg, Real Time Learning

'''
import argparse
import mlflow
from sklearn.linear_model import Ridge, ElasticNet
from xgboost import XGBRegressor
from sklearn.model_selection import ParameterGrid
from data import X_train, X_val, y_train, y_val
from params import ridge_param_grid, elasticnet_param_grid, xgb_param_grid
from eval import eval_metrics


def train(model_name):
    
    # Select model and parameter grid based on input argument
    if model_name == 'ElasticNet':
        model_cls = ElasticNet
        param_grid = elasticnet_param_grid
    elif model_name == 'Ridge':
        model_cls = Ridge
        param_grid = ridge_param_grid
    else:  # Defaults to XGBRegressor if --model is not provided or is incorrect
        model_cls = XGBRegressor
        param_grid = xgb_param_grid

    # NOTE: We usually don't hard-code any experiment/run name/ids,
    # these are set dynamically
    # and here MLproject contains the important configuration,
    # so that the script/code is generic!
    # Also, it is common to have a run.py file which uses hard-coded values (i.e., URIs).
    # Loop through the hyperparameter combinations and log results in separate runs
    for params in ParameterGrid(param_grid):
        with mlflow.start_run():
            # Fit model
            model = model_cls(**params)
            model.fit(X_train, y_train)

            # Evaluate trained model
            y_pred = model.predict(X_val)
            metrics = eval_metrics(y_val, y_pred)

            # Logging the inputs and parameters
            mlflow.log_params(params)
            mlflow.log_metrics(metrics)

            # Log the trained model
            mlflow.log_model(
                model,
                model_name,
                input_example=X_train[:5],
                code_paths=['train.py', 'data.py', 'params.py', 'eval.py']
            )

if __name__ == "__main__":
    # Parse arguments with a default model
    parser = argparse.ArgumentParser(description='Train a model.')
    parser.add_argument('--model', type=str, choices=['ElasticNet', 'Ridge', 'XGBRegressor'], default='ElasticNet', help='The model to train. Defaults to ElasticNet.')
    args = parser.parse_args()

    train(args.model)
