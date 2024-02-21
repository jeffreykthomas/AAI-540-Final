import json
import pathlib
import tarfile
import tensorflow as tf
from transformers import TFBertForSequenceClassification, BertTokenizer
import numpy as np
import os
from sklearn.metrics import f1_score

base_dir = '/opt/ml/processing'


def extract_model(model_path=f'{base_dir}/model/model.tar.gz', extract_path=f'{base_dir}/model/extracted_model'):
    with tarfile.open(model_path) as tar:
        tar.extractall(path=extract_path)
    if not os.path.exists(extract_path):
        os.makedirs(extract_path)
    return extract_path


def load_model(model_dir='model'):
    model = TFBertForSequenceClassification.from_pretrained(model_dir)
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
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
    model_path = f'{base_dir}/model'
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    test_dataset = load_dataset(f'{base_dir}/input/tokenized/test/tokenized_data.tfrecord')
    test_dataset = test_dataset.batch(27).prefetch(tf.data.experimental.AUTOTUNE)
    
    model_extracted_path = extract_model()
    model = load_model(model_extracted_path)
   
    # Evaluate the model
    logits = model.predict(test_dataset).logits
    predictions = np.argmax(logits, axis=1)
    
    true_labels = []
    for _, labels in test_dataset:
        true_labels.extend(labels.numpy())
    true_labels = np.array(true_labels)

    f1 = f1_score(true_labels, predictions, average='weighted')
    accuracy = np.mean(np.array(predictions) == np.array(true_labels))
    
    report_dict = {
        'classification_metrics': {
            'accuracy': {'value': accuracy},
            'f1': {'value': f1},
        },
    }

    output_dir = f'{base_dir}/evaluation'
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

    evaluation_path = f'{output_dir}/evaluation.json'
    with open(evaluation_path, 'w') as f:
        f.write(json.dumps(report_dict))
