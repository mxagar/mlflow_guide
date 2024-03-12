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