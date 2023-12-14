from utils.encoder.model.Models import LogEncoder
from utils.encoder.model.Encoder import LSTMEncoder
from utils.encoder.dataset.dataset import LogEncodingWithLabelDataset, collate_fn_label
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import torch
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def print_label(labels):
    normal = len(labels[labels == 0])
    anomalous = len(labels[labels == 1])
    total = len(labels)
    print(f'normal:{normal},anomalous:{anomalous},total:{total}')




def training_encoding(s_log_seqs, s_labels, t_log_seqs, t_labels, config):
    model = LSTMEncoder(config['d_input'], config['d_output'], 2)
    # model.load_state_dict(torch.load(model_path))
    model.to(device)
    t_index = np.arange(len(t_log_seqs))
    t_normal_index = t_index[t_labels == 0]
    t_abnormal_index = t_index[t_labels == 1]
    np.random.shuffle(t_normal_index)
    np.random.shuffle(t_abnormal_index)
    ratio = config['num'] / len(t_log_seqs)
    t_normal_num = int(len(t_normal_index) * ratio)
    t_abnormal_num = config['num'] - t_normal_num
    if t_abnormal_num < config['min_num']:
        t_abnormal_num = config['min_num']
        t_normal_num = config['num'] - config['min_num']

    t_training_index = np.concatenate([t_normal_index[:t_normal_num], t_abnormal_index[:t_abnormal_num]])

    t_log_seqs = t_log_seqs[t_training_index]
    t_labels = t_labels[t_training_index]

    # gengrate dataset
    log_seqs = np.concatenate([s_log_seqs, t_log_seqs])
    # log_classes = np.concatenate([np.array([0]*len(s_log_seqs)),np.array([1]*len(t_log_seqs))])
    log_labels = np.concatenate([s_labels, t_labels])
    print_label(s_labels)
    print_label(t_labels)

    index = np.arange(len(log_seqs))
    np.random.shuffle(index)

    log_seqs = log_seqs[index]
    # log_classes=log_classes[index]
    log_labels = log_labels[index]

    # dataset = TensorDataset(torch.FloatTensor(log_seqs),torch.Tensor(log_classes),torch.Tensor(log_labels))
    dataset = TensorDataset(torch.FloatTensor(log_seqs), torch.Tensor(log_labels))
    train_loader = DataLoader(dataset, batch_size=config['batch_size'])
    #
    epoch = config['epoch']

    criterion1 = nn.CrossEntropyLoss()
    criterion2 = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=5e-6)
    iter = tqdm(range(epoch))
    for i in iter:
        total_loss = 0
        frep = 0
        model.train()
        for _, (seq, lab) in enumerate(train_loader):
            torch.cuda.empty_cache()
            seq = seq.to(device)
            lab = lab.to(device).to(torch.long)
            # cla = cla.to(device).to(torch.long)
            _, out1 = model(seq)
            loss1 = criterion1(out1, lab)

            optimizer.zero_grad()
            loss = loss1
            loss.backward()
            optimizer.step()
            total_loss += loss
            frep += 1
        avg_loss = total_loss / frep
        iter.set_postfix(loss='{:.6f}'.format(avg_loss))

    return model



def encoding(log_seqs, config, encoder):
    encoder.to(device)
    encoder.eval()
    log_seqs = torch.tensor(log_seqs).to(device)
    log_vectors = []
    batch_size = config['batch_size']
    for i in range(0, len(log_seqs), batch_size):
        batch_log_seqs = log_seqs[i:i + batch_size]
        batch_log_vectors, _ = encoder(batch_log_seqs)
        log_vectors.append(batch_log_vectors.cpu().detach().numpy())
    log_vectors = np.concatenate(log_vectors)
    return log_vectors



def generate_log_vectors(s_log_seqs, s_log_labels, t_log_seqs, t_log_labels, config):
    source = config['dataset']["source_dataset_name"]
    target = config['dataset']["target_dataset_name"]

    # log vectors and labels path
    s_log_vectors_path = f'./dataset/{source}/{source}_to_{target}_log_vectors.npy'
    s_log_vectors_labels_path = f'./dataset/{source}/{source}_to_{target}_log_vectors_labels.npy'
    t_log_vectors_path = f'./dataset/{target}/{source}_to_{target}_log_vectors.npy'
    t_log_vectors_labels_path = f'./dataset/{target}/{source}_to_{target}_log_vectors_labels.npy'

    s_index = np.arange(len(s_log_seqs))
    t_index = np.arange(len(t_log_seqs))

    s_normal_index = s_index[s_log_labels == 0]
    s_abnormal_index = s_index[s_log_labels == 1]
    t_normal_index = t_index[t_log_labels == 0]
    t_abnormal_index = t_index[t_log_labels == 1]

    encode_ratio = config['encoder']['ratio']
    s_e_len = int((len(s_index)) * encode_ratio)
    s_e_normal_len = int(len(s_normal_index) * encode_ratio)
    s_e_abnormal_len = s_e_len - s_e_normal_len
    t_e_len = int((len(t_index)) * encode_ratio)
    t_e_normal_len = int(len(t_normal_index) * encode_ratio)
    t_e_abnormal_len = t_e_len - t_e_normal_len

    s_e_index = np.concatenate([s_normal_index[:s_e_normal_len], s_abnormal_index[:s_e_abnormal_len]])
    t_e_index = np.concatenate([t_normal_index[:t_e_normal_len], t_abnormal_index[:t_e_abnormal_len]])
    s_index = np.concatenate([s_normal_index[s_e_normal_len:], s_abnormal_index[s_e_abnormal_len:]])
    t_index = np.concatenate([t_normal_index[t_e_normal_len:], t_abnormal_index[t_e_abnormal_len:]])

    s_e_training = s_log_seqs[s_e_index]
    s_e_label = s_log_labels[s_e_index]
    t_e_training = t_log_seqs[t_e_index]
    t_e_label = t_log_labels[t_e_index]

    s_log_seqs = s_log_seqs[s_index]
    s_log_labels = s_log_labels[s_index]
    t_log_seqs = t_log_seqs[t_index]
    t_log_labels = t_log_labels[t_index]

    print(f'training encoder: training size:{len(s_e_training)}')

    if not config['global']['use_transfer_learning']:
        return s_log_seqs, s_log_labels, t_log_seqs, t_log_labels

    encoder = training_encoding(s_e_training, s_e_label, t_e_training, t_e_label, config['encoder'])

    s_log_vectors = encoding(s_log_seqs, config['encoder'], encoder)
    t_log_vectors = encoding(t_log_seqs, config['encoder'], encoder)

    np.save(s_log_vectors_path, s_log_vectors)
    np.save(s_log_vectors_labels_path, s_log_labels)
    np.save(t_log_vectors_path, t_log_vectors)
    np.save(t_log_vectors_labels_path, t_log_labels)

    return s_log_vectors, s_log_labels, t_log_vectors, t_log_labels
