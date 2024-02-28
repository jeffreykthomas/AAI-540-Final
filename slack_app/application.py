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

    sagemaker_client = boto3.client('sagemaker-runtime', region_name='us-east-2')
    payload = json.dumps({"text": message_text})
    logger.info(f"Invoking SageMaker endpoint with payload: {payload}")

    response = sagemaker_client.invoke_endpoint(
        EndpointName='emotion-ai-endpoint',
        ContentType='application/json',
        Body=payload
    )
    label_to_emoji = {
        "anger": ":angry:",
        "disgust": ":face_vomiting:",
        "fear": ":fearful:",
        "happy": ":smiley:",
        "optimistic": ":grinning_face_with_star_eyes:",
        "affectionate": ":heart_eyes:",
        "sad": ":cry:",
        "surprised": ":open_mouth:",
        "neutral": ":neutral_face:"
    }
    response_body = response['Body'].read().decode('utf-8')
    if response_body:
        inference_result = json.loads(response_body)
        logger.info(f"Inference results: {inference_result}")
        inference_result = inference_result[0]
        logger.info(f"Received inference result from SageMaker model: {inference_result}")
        inference_dict = json.loads(inference_result)
        # round the probabilities to 2 decimal places
        inference_result = {key: round(float(value), 2) for key, value in inference_dict.items()}
        # Convert the labels to emojis
        inference_result = {label_to_emoji[key]: value for key, value in inference_result.items()}
        inference_string = ', '.join(f"{key}: {value*100}%" for key, value in inference_result.items())
        # Respond in Slack using ephemeral message
        result_message = f"*_Message to analyze_*: {message_text}\n*_Prediction result_*: {inference_string}"
        client.chat_postEphemeral(
            channel=shortcut['channel']['id'],
            user=shortcut['user']['id'],
            text=result_message
        )
        logger.info(f"Successfully sent message to Slack: {result_message}")


# Start your app
if __name__ == "__main__":
    logger.info("Starting Slack app in Socket Mode")
    SocketModeHandler(app, SLACK_APP_TOKEN).start()

