'''Housing price prediction project.
MLflow is used for tracking and the project is deployed on AWS.

This script runs the entrypoints defined in MLproject.

Original code from the Udemy course

    MLflow in Action - Master the art of MLOps using MLflow tool
    https://www.udemy.com/course/mlflow-course/

    Author: J Garg, Real Time Learning

'''
import mlflow

experiment_name = "ElasticNet"
entry_point = "Training"

mlflow.set_tracking_uri("http://127.0.0.1:5000")

mlflow.projects.run(
    uri=".",
    entry_point=entry_point,
    experiment_name=experiment_name,
    env_manager="conda"
)
