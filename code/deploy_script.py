import boto3
import json
import time

def lambda_handler(event, context):
    current_time = time.strftime("%m-%d-%H-%M-%S", time.localtime())
    client = boto3.client('sagemaker')
    
    model_name = event['model_name']
    endpoint_config_name = f'{event["endpoint_config_name"]}-{current_time}'
    endpoint_name = event['endpoint_name']
    
    instance_type = event['endpoint_instance_type']

    # Create an endpoint configuration
    endpoint_config_response = client.create_endpoint_config(
        EndpointConfigName=endpoint_config_name,
        ProductionVariants=[
            {
                'VariantName': 'AllTraffic',
                'ModelName': model_name,
                'InitialInstanceCount': 1,
                'InstanceType': instance_type,
            }
        ]
    )
    print(f'Endpoint Config Arn: {endpoint_config_response["EndpointConfigArn"]}')

    list_endpoints_response = client.list_endpoints(
        SortBy="CreationTime",
        SortOrder="Descending",
        NameContains=endpoint_name,
    )
    print(f"list_endpoints_response: {list_endpoints_response}")

    if len(list_endpoints_response["Endpoints"]) > 0:
        print("Updating Endpoint with new Endpoint Configuration")
        update_endpoint_response = client.update_endpoint(
            EndpointName=endpoint_name, EndpointConfigName=endpoint_config_name
        )
        print(f"update_endpoint_response: {update_endpoint_response}")
    else:
        print("Creating Endpoint")
        create_endpoint_response = client.create_endpoint(
            EndpointName=endpoint_name, EndpointConfigName=endpoint_config_name
        )
        print(f"create_endpoint_response: {create_endpoint_response}")
    
    return {'statusCode': 200, 'body': json.dumps('Endpoint Created Successfully')}
