'''Housing price prediction project.
MLflow is used for tracking and the project is deployed on AWS.

This script runs the entrypoints defined in MLproject.

Original code from the Udemy course

    MLflow in Action - Master the art of MLOps using MLflow tool
    https://www.udemy.com/course/mlflow-course/

    Author: J Garg, Real Time Learning

'''
import mlflow

models = ["ElasticNet", "Ridge", "XGBRegressor"]
entry_point = "main"

# We will change this depending on local tests / AWS runs
#mlflow.set_tracking_uri("http://ec2-<IP-number>.<region>.amazonaws.com:5000")
mlflow.set_tracking_uri("http://127.0.0.1:5000")

for model in models:
    experiment_name = model
    mlflow.set_experiment(experiment_name)
    
    mlflow.projects.run(
        uri=".",
        entry_point=entry_point,
        parameters={"model": model},
        env_manager="conda"
    )
