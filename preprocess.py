import pandas as pd
import json, pickle, os
from tqdm import tqdm
import random


def get_data(path, train=True):
    headers = ['contract_id', 'month', 'disconnect_count', 'usage', 'payment', 'total_call_count', \
            'problem_call_count', 'traffic_tag', 'labels']
    if not train:
        headers.remove("labels")
    data = pd.read_csv(path, header=None, names=headers)
    return data

def normalize_data(data, config, train=True):
    cols = list(data.columns)
    cols.remove('contract_id')
    cols.remove('month')
    if train:
        cols.remove('labels')
        metadata = {}
    else:
        with open(config["train"]["metadata_path"], "r") as f:
            metadata = json.load(f)
    for col in cols:
        if train:
            mean, std = data[col].mean(), data[col].std(ddof=0)
            metadata[col] = {"mean":mean, "std":std}
        else:
            mean, std = metadata[col]["mean"], metadata[col]["std"]
        data[col] = (data[col] - mean)/std
    if train:
        with open(config["train"]["metadata_path"], "w") as f:
            json.dump(metadata, f)
    return data

def postprocess(data, train=True):
    new_data, labels, contract_ids = {}, {}, []
    grouped_data = data.groupby('contract_id')
    for name, group in tqdm(grouped_data):
        group.sort_values('month')
        group.pop('month')
        group.pop('contract_id')
        if train:
            label = group["labels"].tolist()[-1]
            group.pop('labels')
            labels[name] = label
        new_data[name] = group.values
        contract_ids.append(name)
    if train:
        return new_data, labels, contract_ids
    else:
        return new_data, contract_ids

def split_data(data, labels, contract_ids, val_size):
    random.shuffle(contract_ids)
    split_index = int(len(contract_ids)*val_size)
    val_contract_ids = contract_ids[:split_index]
    train_contract_ids = contract_ids[split_index:]
    train_data = {contract_id:data[contract_id] for contract_id in train_contract_ids}
    train_labels = {contract_id:labels[contract_id] for contract_id in train_contract_ids}
    val_data = {contract_id:data[contract_id] for contract_id in val_contract_ids}
    val_labels = {contract_id:labels[contract_id] for contract_id in val_contract_ids}
    train_dataset = {"data":train_data, "labels":train_labels, "contract_ids":train_contract_ids}
    val_dataset = {"data":val_data, "labels":val_labels, "contract_ids":val_contract_ids}
    return train_dataset, val_dataset
    
def save_processed_data(data, path):
    with open(path, 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

def preprocess(config):    
    print("Preprocess train data ...")
    data = get_data(config["train"]["train_data_path"], True)
    data = normalize_data(data, config, True)
    data, labels, contract_ids = postprocess(data, True)
    train_dataset, val_dataset = split_data(data, labels, contract_ids, config["train"]["val_size"])
    save_processed_data(train_dataset, config["train"]["processed_train_data_path"])
    save_processed_data(val_dataset, config["train"]["processed_val_data_path"])

    print("Preprocess test data ...")
    test_data = get_data(config["test"]["test_data_path"], False)
    test_data = normalize_data(test_data, config, False)
    test_data, test_contract_ids = postprocess(test_data, False)
    test_dataset = {"data":test_data, "contract_ids":test_contract_ids}
    save_processed_data(test_dataset, config["test"]["processed_test_data_path"])


if __name__ == "__main__":
    import yaml
    config = yaml.load(open("config.yaml", "r"), Loader=yaml.FullLoader)
    preprocess(config)
