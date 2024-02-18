
import tensorflow as tf
import json
from transformers import BertTokenizer, TFBertForSequenceClassification, AutoConfig


def model_fn(model_dir):
	config = AutoConfig.from_pretrained(model_dir)
	model = TFBertForSequenceClassification.from_pretrained(model_dir, config=config)
	return model


def input_fn(request_body, request_content_type):
	# Process and tokenize the incoming request
	if request_content_type == 'application/json':
		input_data = json.loads(request_body)
		raw_text = input_data['text']
		# Tokenize the input text
		tokenizer = BertTokenizer.from_pretrained('/opt/ml/model')
		tokens = tokenizer.encode_plus(raw_text, truncation=True, padding=True, max_length=50, return_tensors='tf')
		return {'input_ids': tokens['input_ids'], 'attention_mask': tokens['attention_mask']}
	else:
		raise ValueError('This model only supports application/json input')


def predict_fn(input_data, model):
	predictions = model(**input_data)
	return predictions


def logits_to_top_n_prob(logits, class_names, top_n=3):
	if hasattr(logits, 'logits'):
		logits = logits.logits
	# Convert the logits to probabilities
	probabilities = tf.nn.softmax(logits, axis=-1)
	probabilities = probabilities.numpy().flatten()
	# Get the top n probabilities
	top_n_indices = probabilities.argsort()[-top_n:][::-1]
	top_n_probabilities = {class_names[i]: float(probabilities[i]) for i in top_n_indices}

	return top_n_probabilities


def output_fn(prediction, content_type):
	classes = ['anger', 'disgust', 'fear', 'happy', 'optimistic', 'affectionate', 'sad', 'surprised', 'neutral']
	# Convert the logits to probabilities
	probabilities = logits_to_top_n_prob(prediction, classes, top_n=3)
	if content_type == 'application/json':
		return json.dumps(probabilities), 'application/json'
	else:
		raise ValueError('This model only supports application/json output')
