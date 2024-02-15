import argparse
import os
import tensorflow as tf
import pandas as pd
import io
import boto3
import sagemaker
import ast
import time
from transformers import BertTokenizer, TFBertForSequenceClassification, BertConfig


def model_fn(num_labels, dropout_prob):
	config = BertConfig.from_pretrained(
		'bert-base-uncased',
		num_labels=num_labels,
		hidden_dropout_prob=dropout_prob,
		attention_probs_dropout_prob=dropout_prob)
	return TFBertForSequenceClassification.from_pretrained('bert-base-uncased', config=config)


def wait_for_athena_query_completion(client, query_execution_id):
	while True:
		response = client.get_query_execution(QueryExecutionId=query_execution_id)
		status = response['QueryExecution']['Status']['State']
		if status in ['SUCCEEDED', 'FAILED', 'CANCELLED']:
			return status
		time.sleep(5)


if __name__ == '__main__':
	parser = argparse.ArgumentParser()

	# Hyperparameters
	parser.add_argument('--num_layers_to_freeze', type=int, default=0)
	parser.add_argument('--dropout_prob', type=float, default=0.1)
	parser.add_argument('--batch_size', type=int, default=16)
	parser.add_argument('--epochs', type=int, default=3)
	parser.add_argument('--initial_learning_rate', type=float, default=2e-5)
	parser.add_argument('--lr_scheduler', type=str, default='PolynomialDecay')
	parser.add_argument('--decay_steps', type=int, default=5000)
	parser.add_argument('--power_exp', type=float, default=1.0)

	args = parser.parse_args()

	# Load your data here
	region_name = 'us-west-2'
	bucket_name = 'aai-540-final-data'

	boto_session = boto3.Session(region_name=region_name)

	session = sagemaker.Session(boto_session=boto_session)
	featurestore_runtime = session.boto_session.client(
		service_name='sagemaker-featurestore-runtime',
		region_name=region_name)
	feature_group_name = 'emotion_feature_group_13_03_13_24_1707794028'

	athena_client = boto3.client('athena', region_name=region_name)
	query_string = f"""
	SELECT * FROM "{feature_group_name}"
	WHERE data_type = 'train'
    """

	output_location = f's3://{bucket_name}/athena/results/'

	response_train = athena_client.start_query_execution(
		QueryString=query_string,
		QueryExecutionContext={
			'Database': 'sagemaker_featurestore'
		},
		ResultConfiguration={
			'OutputLocation': output_location,
		}
	)
	train_location_id = response_train['QueryExecutionId']

	val_query_string = f"""
	SELECT * FROM "{feature_group_name}"
	WHERE data_type = 'val'
	"""

	response_val = athena_client.start_query_execution(
		QueryString=val_query_string,
		QueryExecutionContext={
			'Database': 'sagemaker_featurestore'
		},
		ResultConfiguration={
			'OutputLocation': output_location,
		}
	)
	val_location_id = response_val['QueryExecutionId']

	status_train = wait_for_athena_query_completion(athena_client, train_location_id)
	status_val = wait_for_athena_query_completion(athena_client, val_location_id)

	if status_train == 'SUCCEEDED' and status_val == 'SUCCEEDED':
		s3 = session.boto_session.client('s3')
		s3_path = 'athena/results/'
		s3_train_location = s3_path + train_location_id + '.csv'
		s3_val_location = s3_path + val_location_id + '.csv'

		train_data_obj = s3.get_object(Bucket=bucket_name, Key=s3_train_location)
		val_data_obj = s3.get_object(Bucket=bucket_name, Key=s3_val_location)

		df_train = pd.read_csv(io.BytesIO(train_data_obj['Body'].read()))
		df_val = pd.read_csv(io.BytesIO(val_data_obj['Body'].read()))
	else:
		raise ValueError("Athena query failed")

	emotion_categories = {
		"anger": ["anger", "annoyance", "disapproval"],
		"disgust": ["disgust"],
		"fear": ["fear", "nervousness"],
		"happy": ["joy", "amusement", "approval", "gratitude"],
		"optimistic": ["optimism", "relief", "pride", "excitement"],
		"affectionate": ["love", "caring", "admiration", "desire"],
		"sadness": ["sadness", "disappointment", "embarrassment", "grief", "remorse"],
		"surprise": ["surprise", "realization", "confusion", "curiosity"],
		"neutral": ["neutral"]
	}
	category_to_index = {category: index for index, category in enumerate(emotion_categories)}


	def feature_store_to_dataset(dataframe, category_to_index, shuffle=True, batch_size=16):
		dataframe = dataframe.copy()

		# Extract labels and convert to numerical values
		labels = dataframe.pop('emotions').apply(lambda x: category_to_index[x]).values

		# Parse 'input_ids' and 'attention_mask' from strings to lists of integers
		input_ids = dataframe['input_ids'].apply(ast.literal_eval).tolist()
		attention_mask = dataframe['attention_mask'].apply(ast.literal_eval).tolist()

		# Convert lists to TensorFlow tensors
		input_ids = tf.constant(input_ids, dtype=tf.int32)
		attention_mask = tf.constant(attention_mask, dtype=tf.int32)

		# Create a TensorFlow dataset
		ds = tf.data.Dataset.from_tensor_slices(({"input_ids": input_ids, "attention_mask": attention_mask}, labels))

		# Shuffle and batch the dataset
		if shuffle:
			ds = ds.shuffle(buffer_size=len(dataframe))
		ds = ds.batch(batch_size)

		return ds


	train_dataset = feature_store_to_dataset(df_train, category_to_index, shuffle=True, batch_size=args.batch_size)
	val_dataset = feature_store_to_dataset(df_val, category_to_index, shuffle=False, batch_size=args.batch_size)

	# Initialize the tokenizer and model
	tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
	model = model_fn(num_labels=9, dropout_prob=args.dropout_prob)

	# Compile and train your model here
	if args.lr_scheduler == 'PolynomialDecay':
		lr_scheduler = tf.keras.optimizers.schedules.PolynomialDecay(
			initial_learning_rate=args.initial_learning_rate,
			decay_steps=args.decay_steps,
			power=args.power_exp
		)
	elif args.lr_scheduler == 'CosineDecay':
		lr_scheduler = tf.keras.experimental.CosineDecay(
			initial_learning_rate=args.initial_learning_rate,
			decay_steps=args.decay_steps
		)
	else:
		raise ValueError("Invalid learning rate scheduler")

	optimizer = tf.keras.optimizers.Adam(learning_rate=lr_scheduler)
	loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

	model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
	model.fit(train_dataset, epochs=args.epochs, validation_data=val_dataset)
	# Save your model to SM_MODEL_DIR
	path = os.environ['SM_MODEL_DIR']
	model.save_pretrained(path)
	tokenizer.save_pretrained(path)

	tar_file = 'tuned_model.tar.gz'
	command = f"tar -czvf {tar_file} -C {path} ."
	os.system(command)
	# Upload the tar file to S3
	s3 = boto3.client('s3')
	s3_key = f'models/{tar_file}'
	s3.upload_file(tar_file, bucket_name, s3_key)
