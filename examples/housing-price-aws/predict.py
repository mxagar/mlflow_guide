'''Housing price prediction project.
MLflow is used for tracking and the project is deployed on AWS.

This module tests the inference.

Original code from the Udemy course

    MLflow in Action - Master the art of MLOps using MLflow tool
    https://www.udemy.com/course/mlflow-course/

    Author: J Garg, Real Time Learning

'''
from data import test
import boto3
import json

endpoint_name = "prod-endpoint"
region = 'us-east-1'

sm = boto3.client('sagemaker', region_name=region)
smrt = boto3.client('runtime.sagemaker', region_name=region)

test_data_json = json.dumps({'instances': test[:20].toarray()[:, :-1].tolist()})

prediction = smrt.invoke_endpoint(
    EndpointName=endpoint_name,
    Body=test_data_json,
    ContentType='application/json'
)

prediction = prediction['Body'].read().decode("ascii")

print(prediction)