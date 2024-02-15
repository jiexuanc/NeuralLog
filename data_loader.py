
import random
import json
import pandas as pd
import numpy as np
from collections import OrderedDict
import re
from sklearn.utils import shuffle
import pickle
import string
from transformers import GPT2Tokenizer, TFGPT2Model
from transformers import BertTokenizer, TFBertModel
from transformers import RobertaTokenizer, TFRobertaModel
import tensorflow as tf
import time
from datetime import datetime

# bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# bert_model = TFBertModel.from_pretrained('bert-base-uncased')
def bert_encoder(s, no_wordpiece=0):
    """ Compute semantic vector with BERT
    Parameters
    ----------
    s: string to encode
    no_wordpiece: 1 if you do not use sub-word tokenization, otherwise 0

    Returns
    -------
        np array in shape of (768,)
    """
    if no_wordpiece:
        words = s.split(" ")
        words = [word for word in words if word in bert_tokenizer.vocab.keys()]
        s = " ".join(words)
    inputs = bert_tokenizer(s, return_tensors='tf', max_length=512)
    outputs = bert_model(**inputs)
    v = tf.reduce_mean(outputs.last_hidden_state, 1)
    return v[0]


# Takes processed and labelled individual logs and do stuff...

def load_darpa(json_file, train_ratio=0.5, split_type='uniform'):
    with open(json_file, mode="r", encoding="utf-8") as file:
        logs = file.readlines()
        logs = [x.strip() for x in logs]
        logs = [json.loads(x) for x in logs]
    print(f"{len(logs)} lines of logs loaded")

    # Split into benign/malicious
    df_benign = pd.DataFrame([event for event in logs if event["label"] == 0])
    df_malicious = pd.DataFrame([event for event in logs if event["label"] == 1])

    mixed_df = mix(df_benign, df_malicious)

    # About 5% of logs are malicious...

# Given 2 dataframes, mix while retaining the original order -- this is something pretty atrocious
def mix(df_benign:pd.DataFrame, df_malicious:pd.DataFrame):
    order_benign = np.sort(np.random.rand(len(df_benign)))
    order_malicious = np.sort(np.random.rand(len(df_malicious)))
    df_benign["order"] = order_benign
    df_malicious["order"] = order_malicious
    mixed_df = pd.concat([df_benign, df_malicious], ignore_index=True)
    mixed_df = mixed_df.sort_values(by="order").reset_index(drop=True)
    mixed_df.drop("order", axis=1, inplace=True)
    
    return mixed_df

def main():
    load_darpa("201_actual.json")

main()



