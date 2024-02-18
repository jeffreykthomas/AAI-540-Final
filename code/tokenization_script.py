from transformers import BertTokenizer
import pandas as pd
import tensorflow as tf
import os

base_dir = '/opt/ml/processing'


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
        encoding = tokenizer.encode_plus(text, truncation=True, padding=True, max_length=50)
        serialized_example = serialize_example(encoding['input_ids'], encoding['attention_mask'], label)
        serialized_examples.append(serialized_example)
    
    with tf.io.TFRecordWriter(f'{base_dir}/{destination}/tokenized_data.tfrecord') as writer:
        for example in serialized_examples:
            writer.write(example)

            
if __name__ == '__main__':
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    df_train = pd.read_csv(f'{base_dir}/output/train/train.csv')
    df_val = pd.read_csv(f'{base_dir}/output/validation/validation.csv')
    df_test = pd.read_csv(f'{base_dir}/output/test/test.csv')

    output_dirs = ['output/tokenized/train', 'output/tokenized/validation', 'output/tokenized/test']
    for output_dir in output_dirs:
        full_dir = f'{base_dir}/{output_dir}'
        if not os.path.exists(full_dir):
            os.makedirs(full_dir)

    save_data(df_train['text'].tolist(), df_train['emotions'].tolist(), tokenizer, 'output/tokenized/train')
    save_data(df_val['text'].tolist(), df_val['emotions'].tolist(), tokenizer, 'output/tokenized/validation')
    save_data(df_test['text'].tolist(), df_test['emotions'].tolist(), tokenizer, 'output/tokenized/test')
