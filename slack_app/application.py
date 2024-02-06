import os
from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler
import boto3
import json

# Initialize the app
app = App(token=os.environ.get("SLACK_BOT_TOKEN"), app_token=os.environ.get("SLACK_APP_TOKEN"))


# Handle a message shortcut
@app.shortcut("analyze_message")
def handle_shortcut(ack, shortcut, client):
	ack()

	message_text = shortcut['message']['text']

	sagemaker_client = boto3.client('sagemaker-runtime', region_name='us-west-2')
	response = sagemaker_client.invoke_endpoint(
		EndpointName='emotion-ai-endpoint',
		ContentType='application/json',
		Body=json.dumps({"text": [message_text]})
	)
	inference_result = json.load(response['Body'])

	# Respond in Slack
	client.chat_postMessage(channel=shortcut['user']['id'], text=f"Prediction result: {inference_result}")


# Start your app
if __name__ == "__main__":
	SocketModeHandler(app, os.environ["SLACK_APP_TOKEN"]).start()
