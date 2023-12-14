import yaml
import os
import pandas as pd
import numpy as np

from utils.utils import set_seed
from utils import preprocessing, SlidingWindow
from utils.encoder.encoding import generate_log_vectors
from utils.anomaly_detection.anomaly_detection import LogAction, LogAction_without_transfer

import json
import torch
import argparse


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def pre_process(config):
    s_log_structured_path = './{}/{}/{}.log_structured.csv'.format(
        config['dir'], config['source_dataset_name'],
        config['source_dataset_name'])
    t_log_structured_path = './{}/{}/{}.log_structured.csv'.format(
        config['dir'], config['target_dataset_name'],
        config['target_dataset_name'])
    if not os.path.exists(s_log_structured_path):
        preprocessing.parsing(config['source_dataset_name'], config['dir'])
    if not os.path.exists(t_log_structured_path):
        preprocessing.parsing(config['target_dataset_name'], config['dir'])

    df_source = pd.read_csv(s_log_structured_path)
    print(f'Reading source dataset: {config["source_dataset_name"]}')
    df_target = pd.read_csv(t_log_structured_path)
    print(f'Reading target dataset: {config["target_dataset_name"]}')

    return SlidingWindow.get_datasets_bart(df_source, df_target, config)


def main():
    parser = argparse.ArgumentParser(description="TALog")
    parser.add_argument("--config", type=str, default="thu_to_zoo.yaml")
    args = parser.parse_args()
    path = "config/" + args.config
    with open(path, "r") as f:
        config = yaml.safe_load(f)
    print(json.dumps(config, indent=4))

    seed = config['global']['random_seed']
    set_seed(seed)
    source = config['dataset']["source_dataset_name"]
    target = config['dataset']["target_dataset_name"]

    # log seqs and labels path
    s_log_seqs_path = f'./dataset/{source}/{source}_to_{target}_log_seqs.npy'
    s_log_labels_path = f'./dataset/{source}/{source}_to_{target}_log_labels.npy'
    t_log_seqs_path = f'./dataset/{target}/{source}_to_{target}_log_seqs.npy'
    t_log_labels_path = f'./dataset/{target}/{source}_to_{target}_log_labels.npy'

    # log vectors and labels path
    s_log_vectors_path = f'./dataset/{source}/{source}_to_{target}_log_vectors.npy'
    s_log_vectors_labels_path = f'./dataset/{source}/{source}_to_{target}_log_vectors_labels.npy'
    t_log_vectors_path = f'./dataset/{target}/{source}_to_{target}_log_vectors.npy'
    t_log_vectors_labels_path = f'./dataset/{target}/{source}_to_{target}_log_vectors_labels.npy'

    if not os.path.exists(s_log_labels_path):
        config['global']['need_preprocess'] = True

    if not os.path.exists(s_log_vectors_path):
        config['global']['need_encoding'] = True

    if config['global']['need_encoding']:
        if config['global']['need_preprocess']:
            pre_process(config['dataset'])
        s_log_seqs = np.load(s_log_seqs_path)
        s_log_labels = np.load(s_log_labels_path)
        t_log_seqs = np.load(t_log_seqs_path)
        t_log_labels = np.load(t_log_labels_path)

        s_log_vectors, s_log_labels, t_log_vectors, t_log_labels = generate_log_vectors(s_log_seqs,
                                                                                        s_log_labels,
                                                                                        t_log_seqs,
                                                                                        t_log_labels,
                                                                                        config)


    else:
        s_log_vectors = np.load(s_log_vectors_path)
        s_log_labels = np.load(s_log_vectors_labels_path)
        t_log_vectors = np.load(t_log_vectors_path)
        t_log_labels = np.load(t_log_vectors_labels_path)



    s_ratio = config['anomaly_detection']['s_ratio']
    s_train_size = int(len(s_log_vectors) * s_ratio)

    s_index = np.arange(len(s_log_labels))
    s_normal_index = s_index[s_log_labels == 0]
    s_abnormal_index = s_index[s_log_labels == 1]
    s_train_normal_len = int(len(s_normal_index) * s_ratio)
    s_train_abnormal_len = s_train_size - s_train_normal_len

    s_train_index = np.concatenate([s_normal_index[:s_train_normal_len], s_abnormal_index[:s_train_abnormal_len]])
    s_test_index = np.concatenate([s_normal_index[s_train_normal_len:], s_abnormal_index[s_train_abnormal_len:]])

    s_train_log_vectors = s_log_vectors[s_train_index]
    s_train_log_labels = s_log_labels[s_train_index]
    s_test_log_vectors = s_log_vectors[s_test_index]
    s_test_log_labels = s_log_labels[s_test_index]

    t_ratio = config['anomaly_detection']['t_ratio']

    t_pool_size = int(len(t_log_vectors) * t_ratio)

    t_index = np.arange(len(t_log_labels))
    t_normal_index = t_index[t_log_labels == 0]
    t_abnormal_index = t_index[t_log_labels == 1]
    t_pool_normal_len = int(len(t_normal_index) * t_ratio)
    t_pool_abnormal_len = t_pool_size - t_pool_normal_len

    t_pool_index = np.concatenate([t_normal_index[:t_pool_normal_len], t_abnormal_index[:t_pool_abnormal_len]])
    t_test_index = np.concatenate([t_normal_index[t_pool_normal_len:], t_abnormal_index[t_pool_abnormal_len:]])

    t_pool_log_vectors = t_log_vectors[t_pool_index]
    t_pool_log_labels = t_log_labels[t_pool_index]
    t_test_log_vectors = t_log_vectors[t_test_index]
    t_test_log_labels = t_log_labels[t_test_index]

    if config['global']['use_transfer_learning']:
        LogAction(s_train_log_vectors, s_train_log_labels, s_test_log_vectors, s_test_log_labels,
                  t_pool_log_vectors, t_pool_log_labels, t_test_log_vectors, t_test_log_labels,
                  config['anomaly_detection'])
    else:
        LogAction_without_transfer(s_test_log_vectors, s_test_log_labels,
                                   t_pool_log_vectors, t_pool_log_labels, t_test_log_vectors, t_test_log_labels,
                                   config['anomaly_detection'])



if __name__ == '__main__':
    main()
