import random
import json
import pandas as pd
import numpy as np
from sklearn.utils import shuffle
from transformers import BertTokenizer, BertModel

def _split_data(x_data, y_data=None, train_ratio=0, split_type='uniform'):
    """ Split train/test data
    Parameters
    ----------
    x_data: list, set of log sequences (in the type of semantic vectors)
    y_data: list, labels for each log sequence
    train_ratio: float, training ratio (e.g., 0.8)
    split_type: `uniform` or `sequential`, which determines how to split dataset. `uniform` means
            to split positive samples and negative samples equally when setting label_file. `sequential`
            means to split the data sequentially without label_file. That is, the first part is for training,
            while the second part is for testing.
    Returns
    -------

    """
    (x_data, y_data) = shuffle(x_data, y_data)
    if split_type == 'uniform' and y_data is not None:
        pos_idx = y_data > 0
        x_pos = x_data[pos_idx]
        y_pos = y_data[pos_idx]
        x_neg = x_data[~pos_idx]
        y_neg = y_data[~pos_idx]
        train_pos = int(train_ratio * x_pos.shape[0])
        train_neg = train_pos
        x_train = np.hstack([x_pos[0:train_pos], x_neg[0:train_neg]])
        y_train = np.hstack([y_pos[0:train_pos], y_neg[0:train_neg]])
        x_test = np.hstack([x_pos[train_pos:], x_neg[train_neg:]])
        y_test = np.hstack([y_pos[train_pos:], y_neg[train_neg:]])
    elif split_type == 'sequential':
        num_train = int(train_ratio * x_data.shape[0])
        x_train = x_data[0:num_train]
        x_test = x_data[num_train:]
        if y_data is None:
            y_train = None
            y_test = None
        else:
            y_train = y_data[0:num_train]
            y_test = y_data[num_train:]
    # Random shuffle
    indexes = shuffle(np.arange(x_train.shape[0]))
    x_train = x_train[indexes]
    if y_train is not None:
        y_train = y_train[indexes]
    return (x_train, y_train), (x_test, y_test)

def bert_encoder(s, no_wordpiece=0, as_numpy=True):
    """ Compute semantic vector with BERT
    Parameters
    ----------
    s: string to encode
    no_wordpiece: 1 if you do not use sub-word tokenization, otherwise 0

    Returns
    -------
        np array in shape of (768,) or tensor
    """
    if no_wordpiece:
        words = s.split(" ")
        words = [word for word in words if word in bert_tokenizer.vocab.keys()]
        s = " ".join(words)
    inputs = bert_tokenizer(s, return_tensors='pt', max_length=512, truncation=True)
    outputs = bert_model(**inputs)
    v = torch.mean(outputs.last_hidden_state, 1)
    if as_numpy:
        return v[0].detach().numpy()
    else:
        return v[0]

# Takes processed and labelled individual logs and do stuff...
def load_from_log(log_file, train_ratio=0.8, split_type='uniform', labelled=True):
    with open(log_file, mode="r", encoding="utf-8") as file:
        logs = file.readlines()
        logs = [x.strip() for x in logs]
        logs = [json.loads(x) for x in logs]

    print("\n\n############# LOG MESSAGE START HERE #############\n\n")
    print(f"{len(logs)} lines of logs loaded")

    # Split into benign/malicious
    if labelled:
        df_benign = pd.DataFrame([event for event in logs if event["label"] == 0])
        df_malicious = pd.DataFrame([event for event in logs if event["label"] == 1])
        print(f"{len(df_malicious)} malicious events found")
        print(f"{len(df_benign)} benign events found")

        data_df, encoding = generate_sequences(df_benign, df_malicious)

        np.savez_compressed("data-bert.npz", data_x=data_df["Sequence"].values,
                            data_y=data_df["Label"].values)
        
        (x_train, y_train), (x_test, y_test) = _split_data(data_df['Sequence'].values,
                                                           data_df['Label'].values, train_ratio, split_type)
        
        print(f"Split complete with ratio: {train_ratio}")

    else:
        print("Unlabelled data not supported yet")
        return
    
    return (x_train, y_train), (x_test, y_test)
    
# Given 2 dataframes, mix while retaining the original order of each label
def mix(df_benign:pd.DataFrame, df_malicious:pd.DataFrame):
    df_benign = df_benign.copy()
    df_malicious = df_malicious.copy()

    order_benign = np.sort(np.random.rand(len(df_benign)))
    order_malicious = np.sort(np.random.rand(len(df_malicious)))

    df_benign["order"] = order_benign
    df_malicious["order"] = order_malicious
    mixed_df = pd.concat([df_benign, df_malicious], ignore_index=True)
    mixed_df = mixed_df.sort_values(by="order").reset_index(drop=True)
    mixed_df.drop("order", axis=1, inplace=True)
    
    return mixed_df

# From a dataframe, extract a continuous subsequence of length n and return new dataframe
def rand_subsequence(df:pd.DataFrame, n):
    start_index = random.randint(0, len(df) - n)
    return df.iloc[start_index:start_index + n]

'''
Generation of sequence of events -- about 5% of logs are malicious
About 5% of logs are malicious...
84295 benign in 6 hours ==> 14000/h
2051 malicious in 3 hours ==> 683/h

Say i take a 10 min bucket,  ~2000 log bucket
'''
# Default parameters: num=100, max_length=7500, event_ratio=0.05, label_ratio=0.5 (balanced)
def generate_sequences(df_benign:pd.DataFrame, df_malicious:pd.DataFrame, num=100, max_length=200, event_ratio=0.05, label_ratio=0.5):
    E = {}
    encoder = bert_encoder
    n_bseq = int(num * (1 - label_ratio))
    n_mseq = num - n_bseq
    final_df = pd.DataFrame({"Sequence": [], "Label": []})

    # Check whether benign and malicious dataframe has enough rows
    max_length = int(min(max_length, len(df_benign) * (1 / 1 - event_ratio), len(df_malicious) * (1 / event_ratio)))

    def encoding(entry):
        if entry not in E:
            E[entry] = encoder(entry)
        return E[entry]

    # Generating malicious sequences...
    for i in range(1, n_mseq + 1):
        # Allows 20% fluctuations of malicious events 
        fluctuations = int(max_length * event_ratio * random.uniform(-0.2, 0.2))
        n_benign = max_length * (1 - event_ratio)
        n_malicious = max_length * event_ratio
        n_benign = int(min(n_benign + fluctuations, len(df_benign))) if n_benign + fluctuations > 0 else n_benign
        n_malicious = int(min(n_malicious - fluctuations, len(df_malicious))) if n_malicious - fluctuations > 0 else n_malicious

        # Generate subsequences and mix
        sub_b = rand_subsequence(df_benign, n_benign)
        sub_m = rand_subsequence(df_malicious, n_malicious)
        mixed = mix(sub_b, sub_m)

        # Apply encoding to input
        malicious_series = mixed['input'].apply(encoding)
        final_df = pd.concat([final_df, pd.DataFrame.from_records([{'Sequence': malicious_series.tolist(), 'Label': 1}])], ignore_index=True)
        if i % 5 == 0:
            print(f"{i} malicious sequence generated")

    for i in range(1, n_bseq + 1):
        # Allows 5% fluctuations of benign events -- i dont want to do padding yet...
        # fluctuations = int(max_length * event_ratio * random.uniform(-0.05, 0))
        fluctuations = 0
        sub_benign = rand_subsequence(df_benign, max_length + fluctuations)
        benign_series = sub_benign['input'].apply(encoding)
        final_df = pd.concat([final_df, pd.DataFrame.from_records([{'Sequence': benign_series.tolist(), 'Label': 0}])], ignore_index=True)
        if i % 5 == 0:
            print(f"{i} benign sequence generated")
            
    print(f"{len(final_df)} sequences have been generated")
    print(f"{len(E)} unique messages are used in these sequences")
    
    return final_df, E

def load_from_npz(npz_file, train_ratio=0.8, split_type='uniform'):
    loaded_data = np.load(npz_file, allow_pickle=True)
    (x_train, y_train), (x_test, y_test) = _split_data(loaded_data['data_x'],
                                                    loaded_data['data_y'], train_ratio, split_type)
    
    return (x_train, y_train), (x_test, y_test)
    
def load_darpa(log_file=None, npz_file=None, train_ratio=0.8, split_type="uniform"):
    if log_file:
        return load_from_log(log_file, train_ratio=train_ratio, split_type=split_type)
    if npz_file:
        return load_from_npz(npz_file, train_ratio=train_ratio, split_type=split_type)

    return None

def main():
    load_darpa(log_file="data/201_FLOW_MESSAGE_actual.json")
    # load_darpa(npz_file="data-bert.npz")

if __name__ == "__main__":
    import torch
    # Using SecBERT instead...
    bert_tokenizer = BertTokenizer.from_pretrained("jackaduma/SecBERT")
    bert_model = BertModel.from_pretrained("jackaduma/SecBERT")

    # bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    # bert_model = BertModel.from_pretrained('bert-base-uncased')
    main()



