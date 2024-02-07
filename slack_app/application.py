import os
import logging
import json
import boto3
from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

SLACK_BOT_TOKEN = os.environ.get("SLACK_BOT_TOKEN")
SLACK_APP_TOKEN = os.environ.get("SLACK_APP_TOKEN")

if not SLACK_BOT_TOKEN or not SLACK_APP_TOKEN:
    logger.error("SLACK tokens not found in environment variables")
    raise ValueError("SLACK tokens not found")

# Initialize the app
app = App(token=SLACK_BOT_TOKEN)


# Handle a message shortcut
@app.shortcut("analyze_message")
def handle_shortcut(ack, shortcut, client):
    ack()

    message_text = shortcut['message']['text']
    logger.info(f"Received shortcut request with message: {message_text}")

    sagemaker_client = boto3.client('sagemaker-runtime', region_name='us-west-2')
    payload = json.dumps({"text": message_text})
    logger.info(f"Invoking SageMaker endpoint with payload: {payload}")

    response = sagemaker_client.invoke_endpoint(
        EndpointName='emotion-ai-endpoint',
        ContentType='application/json',
        Body=payload
    )
    inference_result = json.load(response['Body'][0])
    rounded_result = {label: round(percentage * 100, 2) for label, percentage in inference_result.items()}
    logger.info(f"Received inference result from SageMaker: {rounded_result}")

    # Respond in Slack
    result_message = f"Prediction result: {rounded_result}"
    client.chat_postMessage(channel=shortcut['channel']['id'], text=result_message)
    logger.info(f"Successfully sent message to Slack: {result_message}")


# Start your app
if __name__ == "__main__":
    logger.info("Starting Slack app in Socket Mode")
    SocketModeHandler(app, SLACK_APP_TOKEN).start()

