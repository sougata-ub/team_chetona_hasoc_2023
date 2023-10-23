import os.path

import numpy as np
from glob import glob
import re
import json

from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
import utils.stemmer as hindi_stemmer

current_path = os.path.abspath(__file__)
parent_dir = os.path.dirname(current_path)

def tr_flatten(d,lb):
    flat_text = []
    flat_text.append({
        'tweet_id':d['tweet_id'],
        'text':d['tweet'],
        'label':lb[d['tweet_id']],
    })

    for i in d['comments']:
            flat_text.append({
                'tweet_id':i['tweet_id'],
                'text':flat_text[0]['text'] +' [SEP] '+i['tweet'], #flattening comments(appending one after the other)
                'label':lb[i['tweet_id']],
            })
            if 'replies' in i.keys():
                for j in i['replies']:
                    flat_text.append({
                        'tweet_id':j['tweet_id'],
                        'text':flat_text[0]['text'] +' [SEP] '+ i['tweet'] +' [SEP] '+ j['tweet'], #flattening replies
                        'label':lb[j['tweet_id']],
                    })
    return flat_text

def te_flatten(d):
    flat_text = []
    flat_text.append({
        'tweet_id':d['tweet_id'],
        'text':d['tweet'],
    })

    for i in d['comments']:
            flat_text.append({
                'tweet_id':i['tweet_id'],
                'text':flat_text[0]['text'] +' [SEP] ' +i['tweet'],
            })
            if 'replies' in i.keys():
                for j in i['replies']:
                    flat_text.append({
                        'tweet_id':j['tweet_id'],
                        'text':flat_text[0]['text'] +' [SEP] ' + i['tweet'] +' [SEP] ' + j['tweet'],
                    })
    return flat_text

def get_stopwords_stemmer():
    english_stopwords = stopwords.words("english")
    with open(os.path.join(parent_dir, 'final_stopwords.txt'), encoding = 'utf-8') as f:
        hindi_stopwords = f.readlines()
        for i in range(len(hindi_stopwords)):
            hindi_stopwords[i] = re.sub('\n','',hindi_stopwords[i])
    stopword = english_stopwords + hindi_stopwords
    english_stemmer = SnowballStemmer("english")
    return stopword, english_stemmer

def clean_tweet(tweet, english_stemmer, stopword):
    regex_for_english_hindi_emojis="[^a-zA-Z#\U0001F300-\U0001F5FF'|'\U0001F600-\U0001F64F'|'\U0001F680-\U0001F6FF'|'\u2600-\u26FF\u2700-\u27BF\u0900-\u097F]"
    tweet = re.sub(r"@[A-Za-z0-9]+",' ', tweet)
    tweet = re.sub(r"https?://[A-Za-z0-9./]+", ' ', tweet)
    tweet = re.sub(regex_for_english_hindi_emojis,' ', tweet)

    #brackets for [SEP] token are removed we put them back here:
    tweet = re.sub(r" SEP ",'[SEP]', tweet)

    tweet = re.sub("RT ", " ", tweet)
    tweet = re.sub("\n", " ", tweet)
    tweet = re.sub(r" +", " ", tweet)
    tokens = []
    for token in tweet.split():
        if token not in stopword:
            token = english_stemmer.stem(token)
            token = hindi_stemmer.hi_stem(token)
            tokens.append(token)
    return " ".join(tokens)


def get_data(base_addreess, task, mode='train'):
    directories = []

    if mode == 'train' or mode == 'val':
        split = 'train' if mode == 'train' else 'test'

        globs = glob(base_addreess + f"/labeled/2022/{split}/*/")
        if task == 'binary':
            globs += glob(base_addreess + f"/labeled/2021/{split}/*/")
        for i in globs:
            for j in glob(i + '*/'):
                directories.append(j)
        data = []
        for i in directories:
            with open(i + 'data.json', encoding='utf-8') as f:
                data.append(json.load(f))
    else:
        test_files = glob(base_addreess + "/final/*/*")
        data = []
        for i in test_files:
            with open(i, encoding='utf-8') as f:
                data.append(json.load(f))
        test_data = []
        for i in range(len(data)):
            try:
                for j in te_flatten(data[i]):
                    test_data.append(j)
            except:
                print("failed to load data: ", data[i])
                continue
        return test_data


    labels = []
    for i in directories:
        if(task=='binary'):
            try:
                with open(i+'binary_labels.json', encoding='utf-8') as f:
                    labels.append(json.load(f))
            except:
                with open(i+'labels.json', encoding='utf-8') as f:
                    labels.append(json.load(f))
        else:
            with open(i+'contextual_labels.json', encoding='utf-8') as f:
                labels.append(json.load(f))
    data_label = []
    for i in range(len(labels)):
        for j in tr_flatten(data[i], labels[i]):
            data_label.append(j)
    
    files = glob(base_addreess+"/unlabeled/*/*")
    data = []
    for i in files:
        with open(i, encoding='utf-8') as f:
            data.append(json.load(f))
    data_unlabeled = []
    for i in range(len(data)):
        try:
            for j in te_flatten(data[i]):
                data_unlabeled.append(j)
        except:
            continue
    return data_label, data_unlabeled

def get_dataloaders(tweets, labels, tokenizer, val_ratio, batch_size):
    token_id = []
    attention_masks = []

    def preprocessing(input_text, tokenizer):
        return tokenizer.encode_plus(input_text, add_special_tokens = True, max_length = 256, pad_to_max_length = True, return_attention_mask = True, return_tensors = 'pt')

    for sample in tweets:
        encoding_dict = preprocessing(sample, tokenizer)
        token_id.append(encoding_dict['input_ids']) 
        attention_masks.append(encoding_dict['attention_mask'])


    token_id = torch.cat(token_id, dim = 0)
    attention_masks = torch.cat(attention_masks, dim = 0)
    labels = torch.tensor(labels)

    train_idx, val_idx = train_test_split( np.arange(len(labels)), test_size = val_ratio, shuffle = True, stratify = labels)

    train_set = TensorDataset(token_id[train_idx], attention_masks[train_idx], labels[train_idx])

    val_set = TensorDataset(token_id[val_idx], attention_masks[val_idx], labels[val_idx])

    train_dataloader = DataLoader(train_set, sampler = RandomSampler(train_set), batch_size = batch_size)

    validation_dataloader = DataLoader(val_set,sampler = SequentialSampler(val_set),batch_size = batch_size)

    return train_dataloader, validation_dataloader


def get_dataset(tweets, labels, tokenizer, val_ratio, batch_size):
    token_id = []
    attention_masks = []

    def preprocessing(input_text, tokenizer):
        return tokenizer.encode_plus(input_text, add_special_tokens = True, max_length = 256, pad_to_max_length = True, return_attention_mask = True, return_tensors = 'pt')

    for sample in tweets:
        encoding_dict = preprocessing(sample, tokenizer)
        token_id.append(encoding_dict['input_ids'])
        attention_masks.append(encoding_dict['attention_mask'])


    token_id = torch.cat(token_id, dim = 0)
    attention_masks = torch.cat(attention_masks, dim = 0)
    labels = torch.tensor(labels)

    train_idx, val_idx = train_test_split( np.arange(len(labels)), test_size = val_ratio, shuffle = True, stratify = labels)

    train_set = TensorDataset(token_id[train_idx], attention_masks[train_idx], labels[train_idx])

    val_set = TensorDataset(token_id[val_idx], attention_masks[val_idx], labels[val_idx])

    return train_set, val_set
