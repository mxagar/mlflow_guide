'''Housing price prediction project.
MLflow is used for tracking and the project is deployed on AWS.

This module tests the inference.
Before using this script, we must have deployed a model
with deploy.py.

Original code from the Udemy course

    MLflow in Action - Master the art of MLOps using MLflow tool
    https://www.udemy.com/course/mlflow-course/

    Author: J Garg, Real Time Learning

'''
from data import test
import boto3
import json
from dotenv import load_dotenv

# Load environment variables in .env:
# AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_DEFAULT_REGION
load_dotenv()

# Defined in deploy.py
# FIXME: These values should be (environment/repo) variables
endpoint_name = "house-price-prod"
region = 'eu-central-1'

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
