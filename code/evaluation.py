import json
import pathlib
import tarfile
import tensorflow as tf
from transformers import TFBertForSequenceClassification, BertTokenizer
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

base_dir = '/opt/ml/processing'


def extract_model(model_path=f'{base_dir}/model/model.tar.gz', extract_path='model'):
    with tarfile.open(model_path) as tar:
        tar.extractall(path=extract_path)
    return extract_path


def load_model(model_dir='model'):
    model = TFBertForSequenceClassification.from_pretrained(model_dir)
    return model


def _parse_function(proto):
    # Define the feature description dictionary for `tf.io.parse_single_example`
    feature_description = {
        'input_ids': tf.io.FixedLenFeature([50], tf.int64),
        'attention_mask': tf.io.FixedLenFeature([50], tf.int64),
        'label': tf.io.FixedLenFeature([], tf.int64),
    }
    # Parse the input `tf.train.Example` proto using the dictionary above
    parsed_features = tf.io.parse_single_example(proto, feature_description)
    labels = parsed_features.pop('label')
    return parsed_features, labels


def load_dataset(file_path):
    raw_dataset = tf.data.TFRecordDataset(file_path)
    parsed_dataset = raw_dataset.map(_parse_function)
    return parsed_dataset


if __name__ == '__main__':
    model_path = f'{base_dir}/models/tuned_model'
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    test_dataset = load_dataset(f'{base_dir}/output/tokenized/test/tokenized_data.tfrecord')
    test_dataset = test_dataset.batch(16).prefetch(tf.data.experimental.AUTOTUNE)
    
    model_extracted_path = extract_model()
    model = load_model(model_extracted_path)
   
    # Evaluate the model
    loss, accuracy = model.evaluate(test_dataset, return_dict=True)
    predictions = np.argmax(model.predict(test_dataset).logits, axis=1)
    true_labels = np.array([label for _, label in test_dataset])
    
    f1 = f1_score(true_labels, predictions, average='weighted')
    
    report_dict = {
        'classification_metrics': {
            'accuracy': {'value': accuracy},
            'f1_score': {'value': f1},
        },
    }

    output_dir = f'{base_dir}/evaluation'
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

    evaluation_path = f'{output_dir}/evaluation.json'
    with open(evaluation_path, 'w') as f:
        f.write(json.dumps(report_dict))
