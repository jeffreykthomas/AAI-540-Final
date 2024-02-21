import os
import csv
from transformers import BertTokenizer
import tensorflow as tf

base_dir = '/opt/ml/processing'


def load_data_from_csv(file_path):
    texts, labels = [], []
    with open(file_path, newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)  # Skip header
        for row in reader:
            texts.append(row[0])
            labels.append(int(row[1]))  # Ensure label is int if it's not already
    return texts, labels 


def serialize_example(token_ids, attention_mask, label):
    feature = {
        'input_ids': tf.train.Feature(int64_list=tf.train.Int64List(value=token_ids)),
        'attention_mask': tf.train.Feature(int64_list=tf.train.Int64List(value=attention_mask)),
        'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[label]))
    }
    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()


def save_data(texts, labels, tokenizer, destination):
    serialized_examples = []
    for text, label in zip(texts, labels):
        encoding = tokenizer.encode_plus(text, truncation=True, padding='max_length', max_length=50)
        serialized_example = serialize_example(encoding['input_ids'], encoding['attention_mask'], label)
        serialized_examples.append(serialized_example)
    
    with tf.io.TFRecordWriter(f'{destination}/tokenized_data.tfrecord') as writer:
        for example in serialized_examples:
            writer.write(example)

            
if __name__ == '__main__':
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    datasets = ['train', 'validation', 'test']
    
    for dataset in datasets:
        file_path = f'{base_dir}/input/{dataset}/{dataset}.csv'
        texts, labels = load_data_from_csv(file_path)
        output_dir = f'{base_dir}/output/tokenized/{dataset}'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        save_data(texts, labels, tokenizer, output_dir)
