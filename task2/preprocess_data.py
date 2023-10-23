import sys

sys.path.append('..')

import pandas as pd

from utils.data_utils import get_stopwords_stemmer, get_dataloaders, get_data, clean_tweet, get_dataset
from sklearn.preprocessing import LabelEncoder

import os
import json
import nltk

nltk.download('stopwords')

import argparse
parser = argparse.ArgumentParser(description='ICHCL')
parser.add_argument('--data_directory', type=str, default='data', help='Where the dataset is stopred.')
parser.add_argument('--task', type=str, default='binary', help='Binary or multiclass classification.')
parser.add_argument('--mode', type=str, default='train', help='train or test data preprocessing.')
parser.add_argument('--save_data_path', type=str, default=None, help='Path to save the processed data if its not None')


args = parser.parse_args()


def prepare_train_data(data_label, data_unlabeled, args):
    # print(data_label)
    df = pd.DataFrame(data_label, columns = data_label[0].keys(), index = None)
    df['label'].loc[df['label']=='NONE']='NOT'

    print("Number of labeled:", len(df))

    # df_unlabeled = pd.DataFrame(data_unlabeled, columns = data_unlabeled[0].keys(), index = None)
    # print("Number of unlabeled:", len(df_unlabeled))

    print("Binary Distribution")
    print(df['label'].value_counts())
    tweets = df.text
    y = df.label
    stopword, english_stemmer = get_stopwords_stemmer()
    # cleaned_tweets = [clean_tweet(tweet, english_stemmer, stopword) for tweet in tweets]
    cleaned_tweets = tweets
    le = LabelEncoder()
    labels = le.fit_transform(y)
    # print map of label to id
    print(dict(zip(le.classes_, le.transform(le.classes_))))

    # save data as json files
    with open(os.path.join(args.save_data_path), 'w') as f:
        for tweet, label in zip(cleaned_tweets, labels):
            data = {'text': tweet, 'label': int(label)}
            f.write(json.dumps(data) + '\n')


def prepare_test_data(data_test, args):
    df = pd.DataFrame(data_test, columns=data_test[0].keys(), index=None)
    print("Number of test elements:", len(df))

    # save data as json files
    with open(os.path.join(args.save_data_path), 'w') as f:
        for _id, tweet in zip(df.tweet_id, df.text):
            data = {'id': _id, 'text': tweet}
            f.write(json.dumps(data) + '\n')


def main():
    print(args)

    if args.mode == 'train' or args.mode == 'val':
        data_label, data_unlabeled = get_data(args.data_directory, args.task, args.mode)
        prepare_train_data(data_label, data_unlabeled, args)
    else:
        data_test = get_data(args.data_directory, args.task, args.mode)
        prepare_test_data(data_test, args)


if __name__ == '__main__':
    main()