import boto3
import json
from dotenv import load_dotenv
import os

# load environment variables
load_dotenv()
aws_access_key_id = os.getenv('AWS_ACCESS_KEY_ID')
aws_secret_access_key = os.getenv('AWS_SECRET_ACCESS_KEY')
aws_region = os.getenv('AWS_REGION_NAME')


def send_bedrock_command(prompt: str):
    session = boto3.Session(
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key
    )
    bedrock = session.client('bedrock-runtime', region_name=aws_region)
    payload = {
        "inputText": prompt,
        "textGenerationConfig": {
            "maxTokenCount": 1024,
            "stopSequences": [],
            "temperature": 0.7,
            "topP": 1
        }
    }
    body = json.dumps(payload)
    model_id = "amazon.titan-text-premier-v1:0"
    response = bedrock.invoke_model(
        body=body,
        modelId=model_id,
        accept="application/json",
        contentType="application/json",
    )

    response_body = json.loads(response.get("body").read())
    response_text = (response_body.get("results")[0]).get('outputText')
    return response_text

context = """
        Test context
 """

command = f""" 
add command here
context = {context}
"""
result = send_bedrock_command(command)
print(result)
