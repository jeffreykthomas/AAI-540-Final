
import boto3
import csv

import pandas as pd
import io

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

region_name = 'us-west-2'
bucket_name = 'aai-540-final-data'

emotion_labels = [
    'admiration',
    'amusement',
    'anger',
    'annoyance',
    'approval',
    'caring',
    'confusion',
    'curiosity',
    'desire',
    'disappointment',
    'disapproval',
    'disgust',
    'embarrassment',
    'excitement',
    'fear',
    'gratitude',
    'grief',
    'joy',
    'love',
    'nervousness',
    'optimism',
    'pride',
    'realization',
    'relief',
    'remorse',
    'sadness',
    'surprise',
    'neutral'
]

emotion_categories = {
	'anger': ['anger', 'annoyance', 'disapproval'],
	'disgust': ['disgust'],
	'fear': ['fear', 'nervousness'],
	'happy': ['joy', 'amusement', 'approval', 'gratitude'],
	'optimistic': ['optimism', 'relief', 'pride', 'excitement'],
	'affectionate': [ 'love', 'caring', 'admiration',  'desire'],
	'sadness': ['sadness', 'disappointment', 'embarrassment', 'grief',  'remorse'],
	'surprise': ['surprise', 'realization', 'confusion', 'curiosity'],
	'neutral': ['neutral']
}

emotion_to_category = {}
for category, emotions in emotion_categories.items():
	for emotion in emotions:
		emotion_to_category[emotion] = category
        
category_names = list(emotion_categories.keys())
category_to_index = {category: index for index, category in enumerate(category_names)}

if __name__ == '__main__':

    base_dir = '/opt/ml/processing'

    df_1 = pd.read_csv(f'{base_dir}/input/goemotions_1.csv')
    df_2 = pd.read_csv(f'{base_dir}/input/goemotions_2.csv')
    df_3 = pd.read_csv(f'{base_dir}/input/goemotions_3.csv')

    dfs = [df_1, df_2, df_3]

    df_full = pd.concat(dfs, ignore_index=True)

    df_grouped = df_full.groupby('id', as_index=False)[emotion_labels].sum(numeric_only=True)
    df_filtered = df_grouped[(df_grouped[emotion_labels] > 1).any(axis=1)].copy()

    def random_emotion(row):
        emotions_with_agreement = [emotion for emotion in emotion_labels if row[emotion] > 1]
        random.shuffle(emotions_with_agreement)
        return emotions_with_agreement[0] if emotions_with_agreement else None


    df_filtered['selected_emotions'] = df_filtered.apply(random_emotion, axis=1)

    final_df = pd.merge(df_filtered, df_full[['id', 'text']], on='id').drop_duplicates()

    final_df['emotions'] = final_df['selected_emotions'].apply(lambda x: category_to_index.get(emotion_to_category.get(x, 'unknown'), None))
    final_df = final_df.drop(columns=emotion_labels + ['selected_emotions', 'id'])

    # Split the dataset into training and test sets initially
    df_train, df_test = train_test_split(final_df, test_size=0.1, random_state=42)

    # Split the training set further into training and validation sets
    df_train, df_val = train_test_split(df_train, test_size=0.125, random_state=42)

    df_train = df_train.reset_index(drop=True)
    df_val = df_val.reset_index(drop=True)
    df_test = df_test.reset_index(drop=True)

    output_dirs = ['output/train', 'output/validation', 'output/test']
    for output_dir in output_dirs:
        full_dir = f'{base_dir}/{output_dir}'
        if not os.path.exists(full_dir):
            os.makedirs(full_dir)

    df_train.to_csv(f'{base_dir}/output/train/train.csv', index=False)
    df_val.to_csv(f'{base_dir}/output/validation/validation.csv', index=False)
    df_test.to_csv(f'{base_dir}/output/test/test.csv', index=False)
