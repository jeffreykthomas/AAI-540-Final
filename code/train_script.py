import argparse
import os
import tensorflow as tf
from transformers import TFBertForSequenceClassification, BertConfig, BertTokenizer


base_dir = '/opt/ml/processing'


def parse_args():

    parser = argparse.ArgumentParser()

    # hyperparameters sent by the client are passed as command-line arguments to the script
    parser.add_argument('--num_layers_to_freeze', type=int, default=0)
    parser.add_argument('--dropout_prob', type=float, default=0.1)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--initial_learning_rate', type=float, default=2e-5)
    parser.add_argument('--lr_scheduler', type=str, default='PolynomialDecay')
    parser.add_argument('--decay_steps', type=int, default=5000)
    parser.add_argument('--power_exp', type=float, default=1.0)

    # data directories
    parser.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAIN'))
    parser.add_argument('--validation', type=str, default=os.environ.get('SM_CHANNEL_VALIDATION'))

    # model directory
    parser.add_argument('--sm-model-dir', type=str, default=os.environ.get('SM_MODEL_DIR'))

    return parser.parse_known_args()


def model_fn(num_labels, dropout_prob):
    config = BertConfig.from_pretrained(
        'bert-base-uncased',
        num_labels=num_labels,
        hidden_dropout_prob=dropout_prob,
        attention_probs_dropout_prob=dropout_prob)
    return TFBertForSequenceClassification.from_pretrained('bert-base-uncased', config=config)


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
    args, _ = parse_args()
    
    print('Training data location: {}'.format(args.train))
    print('Test data location: {}'.format(args.validation))
    print('Other args: {}'.format(args))
    
    train_dataset = load_dataset(os.path.join(args.train, 'tokenized_data.tfrecord'))
    val_dataset = load_dataset(os.path.join(args.validation, 'tokenized_data.tfrecord'))
    
    train_dataset = train_dataset.shuffle(buffer_size=10000).batch(args.batch_size).prefetch(tf.data.experimental.AUTOTUNE)
    val_dataset = val_dataset.batch(args.batch_size).prefetch(tf.data.experimental.AUTOTUNE)

    # Initialize the model
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
        raise ValueError('Invalid learning rate scheduler')

    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_scheduler)
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
    model.fit(train_dataset, epochs=args.epochs, validation_data=val_dataset)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    model_dir = '/opt/ml/model'
    
    model.save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)