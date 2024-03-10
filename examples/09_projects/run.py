'''Original code from the Udemy course

    MLflow in Action - Master the art of MLOps using MLflow tool
    https://www.udemy.com/course/mlflow-course/

    Author: J Garg, Real Time Learning

Usage:

    conda activate <env_name>
    python ./<filename.py>

'''
import mlflow

parameters={
    "alpha":0.3,
    "l1_ratio":0.3
}

experiment_name = "Project exp 1"
entry_point = "main"

mlflow.projects.run(
    uri=".",
    entry_point=entry_point,
    parameters=parameters,
    experiment_name=experiment_name
)
