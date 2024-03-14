'''Housing price prediction project.
MLflow is used for tracking and the project is deployed on AWS.

This module deploys a model toa SageMaker inference container.

Check this documentation link:

    https://mlflow.org/docs/latest/python_api/mlflow.sagemaker.html#mlflow.sagemaker.SageMakerDeploymentClient.create_deployment

Original code from the Udemy course

    MLflow in Action - Master the art of MLOps using MLflow tool
    https://www.udemy.com/course/mlflow-course/

    Author: J Garg, Real Time Learning

'''
import mlflow.sagemaker
from mlflow.deployments import get_deploy_client

# Specify a unique endpoint name (lower letters)
# FIXME: some the following should be (environment/repo) variables
endpoint_name = "house-price-prod"
# Get model URI from MLflow model registry (hosted on AWS)
model_uri = "..."
# Get from IAM roles
execution_role_arn = "..."
# Get from S3
bucket_name = "..."
# Get from ECR: complete image name with tag
image_url = "..."
flavor = "..."

# Define the missing configuration parameters as a dictionary:
# region, instance type, instance count, etc.
config = {
    "execution_role_arn": execution_role_arn,
    "bucket_name": bucket_name,
    "image_url": image_url,
    "region_name": "eu-central-1", # usually, same as rest of services
    "archive": False,
    "instance_type": "ml.m5.xlarge", # https://aws.amazon.com/sagemaker/pricing/instance-types/
    "instance_count": 1,
    "synchronous": True
}

# Initialize a deployment client for SageMaker
client = get_deploy_client("sagemaker")

# Create the deployment
client.create_deployment(
    name=endpoint_name,
    model_uri=model_uri,
    flavor=flavor,
    config=config,
)
