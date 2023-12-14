import numpy as np
import pandas as pd
from tqdm import tqdm
from nltk.tokenize import RegexpTokenizer
import warnings
from .BART import bart_encode


warnings.filterwarnings("ignore")



def get_sentence_emb(sentence, w2v):
    """
    get a sentence embedding vector
    *automatic initial random value to the new word
    args: sentence(string): sentence of log message
          w2v: word2vec model
    return: sen_emb(list of int): vector for the sentence
    """
    tokenizer = RegexpTokenizer(r'\w+')
    lst = []
    tokens = [x.lower() for x in tokenizer.tokenize(str(sentence))]
    if tokens == []:
        tokens.append('EmptyParametersTokens')
    for i in range(len(tokens)):
        words = w2v.wv.index_to_key
        if tokens[i] in words:
            lst.append(w2v.wv[tokens[i]])
        else:
            w2v.build_vocab([[tokens[i]]], update=True)
            w2v.train([tokens[i]], epochs=1, total_examples=len([tokens[i]]))
            lst.append(w2v.wv[tokens[i]])
    drop = 1
    if len(np.array(lst).shape) >= 2:
        sen_emb = np.mean(np.array(lst), axis=0)
        if len(np.array(lst)) >= 5:
            drop = 0
    else:
        sen_emb = np.array(lst)
    return list(sen_emb), drop



def bart_emb(df):
    corpus = df.EventTemplate.values

    X = bart_encode(corpus)

    df['Embedding'] = [emb for emb in X]

    return df


def sliding_window(df, window_size=20, step_size=4):
    df["Label"] = df["Label"].apply(lambda x: int(x != '-'))
    df = df[["Label", "Embedding"]]
    log_emb_seqs = []
    log_labels = []
    log_size = df.shape[0]
    label_data = df.iloc[:, 0]
    emb_data = []
    for i in df.iloc[:, 1].values:
        emb_data.append(np.array(i))
    for index in tqdm(range(log_size - window_size)):
        if len(log_emb_seqs) > 1000000:
            break
        log_emb_seqs.append(np.array(emb_data[index:index + window_size]))
        log_labels.append(max(label_data[index:index + window_size]))
        index += step_size
    log_emb_seqs = np.array(log_emb_seqs)
    log_labels = np.array(log_labels)
    return log_emb_seqs, log_labels


def sliding_window_HDFS(df, window_size=20, step_size=4):
    log_emb_seqs = []
    log_labels = []
    df = df[["Label", "Embedding", "BlockId"]]
    dict = {}
    label = {}
    for idx, row in df.iterrows():
        if row['BlockId'] not in dict:
            dict[row['BlockId']] = []
            label[row['BlockId']] = row['Label']
        dict[row['BlockId']].append(row['Embedding'])
    for blockid in tqdm(dict):
        if len(log_emb_seqs) > 400000:
            break
        log_size = len(dict[blockid])
        index = 0
        while index <= log_size - window_size:
            log_emb_seqs.append(np.array(dict[blockid][index:index + window_size]))
            log_labels.append(label[blockid])
            index += step_size
    log_emb_seqs = np.array(log_emb_seqs)
    log_labels = np.array(log_labels)
    return log_emb_seqs, log_labels


def sliding_window_Zookeeper(df, window_size=20, step_size=4):
    df["Label"] = df["Level"].apply(lambda x: int(x == 'ERROR'))
    df = df[["Label", "Embedding"]]
    log_emb_seqs = []
    log_labels = []
    log_size = df.shape[0]
    label_data = df.iloc[:, 0]
    emb_data = []
    for i in df.iloc[:, 1].values:
        emb_data.append(np.array(i))
    for index in tqdm(range(log_size - window_size)):
        if len(log_emb_seqs) > 400000:
            break
        log_emb_seqs.append(np.array(emb_data[index:index + window_size]))
        log_labels.append(max(label_data[index:index + window_size]))
        index += step_size
    log_emb_seqs = np.array(log_emb_seqs)
    log_labels = np.array(log_labels)
    return log_emb_seqs, log_labels



def get_datasets_bart(df_source, df_target, config):
    df_source = df_source.iloc[config["s_start"]:config["s_end"]]
    df_target = df_target.iloc[config["t_start"]:config["t_end"]]

    func_dict = {
        'BGL': sliding_window,
        'HDFS': sliding_window_HDFS,
        'Thunderbird': sliding_window,
        'Zookeeper': sliding_window_Zookeeper
    }
    # Get source data preprocessed
    window_size = config["window_size"]
    step_size = config["step_size"]

    source = config["source_dataset_name"]
    target = config["target_dataset_name"]
    emb_dim = config["emb_dim"]

    # save path
    s_log_seqs_path = f'./dataset/{source}/{source}_to_{target}_log_seqs.npy'
    s_log_labels_path = f'./dataset/{source}/{source}_to_{target}_log_labels.npy'
    t_log_seqs_path = f'./dataset/{target}/{source}_to_{target}_log_seqs.npy'
    t_log_labels_path = f'./dataset/{target}/{source}_to_{target}_log_labels.npy'

    # templates path
    s_log_templates_path = './{}/{}/{}.log_templates.csv'.format(
        config['dir'], config['source_dataset_name'],
        config['source_dataset_name'])
    t_log_templates_path = './{}/{}/{}.log_templates.csv'.format(
        config['dir'], config['target_dataset_name'],
        config['target_dataset_name'])
    df_source_templates = pd.read_csv(s_log_templates_path)
    df_target_templates = pd.read_csv(t_log_templates_path)
    all_templates = np.concatenate((df_source_templates.EventTemplate.values, df_target_templates.EventTemplate.values))

    s_log_emb_save_path = './{}/bert_emb/{}.emb.parquet'.format(
        config['dir'], config['source_dataset_name'])
    t_log_emb_save_path = './{}/bert_emb/{}.emb.parquet'.format(
        config['dir'], config['target_dataset_name'])

    if config['need_embedding_source']:
        df_source = bart_emb(df_source)
        df_source.to_parquet(s_log_emb_save_path, index=False)
    else:
        df_source = pd.read_parquet(s_log_emb_save_path)

    if config['need_embedding_target']:
        df_target = bart_emb(df_target)
        df_target.to_parquet(t_log_emb_save_path, index=False)
    else:
        df_target = pd.read_parquet(t_log_emb_save_path)

    print(f'preprocessing for the dataset: {source} and {target}')
    s_log_seqs, s_log_labels = func_dict[source](df_source, window_size, step_size)

    source_max = config['source_max']
    s_index = np.arange(len(s_log_seqs))
    s_normal_index = s_index[s_log_labels == 0]
    s_abnormal_index = s_index[s_log_labels == 1]

    s_ratio = len(s_normal_index) / len(s_index)
    s_normal_len = int(s_ratio * source_max)
    s_abnormal_len = source_max - s_normal_len

    s_normal_index = s_normal_index[:s_normal_len]
    s_abnormal_index = s_abnormal_index[:s_abnormal_len]

    s_index = np.concatenate([s_normal_index, s_abnormal_index])

    s_log_seqs = s_log_seqs[s_index]
    s_log_labels = s_log_labels[s_index]

    np.save(s_log_seqs_path, s_log_seqs)
    np.save(s_log_labels_path, s_log_labels)

    t_log_seqs, t_log_labels = func_dict[target](df_target, window_size, step_size)

    target_max = config['target_max']

    t_index = np.arange(len(t_log_seqs))
    t_normal_index = t_index[t_log_labels == 0]
    t_abnormal_index = t_index[t_log_labels == 1]

    t_ratio = len(t_normal_index) / len(t_index)
    t_normal_len = int(t_ratio * target_max)
    t_abnormal_len = target_max - t_normal_len

    t_normal_index = t_normal_index[:t_normal_len]
    t_abnormal_index = t_abnormal_index[:t_abnormal_len]

    t_index = np.concatenate([t_normal_index, t_abnormal_index])

    t_log_seqs = t_log_seqs[t_index]
    t_log_labels = t_log_labels[t_index]

    np.save(t_log_seqs_path, t_log_seqs)
    np.save(t_log_labels_path, t_log_labels)


