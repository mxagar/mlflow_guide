'''Housing price prediction project.
MLflow is used for tracking and the project is deployed on AWS.

This module the evaluation metric computation.

Original code from the Udemy course

    MLflow in Action - Master the art of MLOps using MLflow tool
    https://www.udemy.com/course/mlflow-course/

    Author: J Garg, Real Time Learning

'''
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error

def eval_metrics(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = mean_absolute_percentage_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    metrics = {
        "RMSE": rmse,
        "MAPE": mape,
        "R2": r2
    }

    return metrics
