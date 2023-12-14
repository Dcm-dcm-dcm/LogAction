from utils.anomaly_detection.model.Models import LogLSTM
from torch.utils.data import TensorDataset, DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np
from utils.anomaly_detection.loss import FreeEnergyAlignmentLoss, NLLLoss
import math
from utils.anomaly_detection.dataset.LogDataset import LogDataset
import torch.nn.functional as F
import random

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def momentum_update(ema, current):
    lambd = np.random.uniform()
    return ema * lambd + current * (1 - lambd)


def print_label(labels):
    normal = len(labels[labels == 0])
    anomalous = len(labels[labels == 1])
    total = len(labels)
    print(f'normal:{normal},anomalous:{anomalous},total:{total}')




def LogAction(s_train_log_vectors, s_train_log_labels, s_test_log_vectors, s_test_log_labels,
              t_pool_log_vectors, t_pool_log_labels, t_test_log_vectors, t_test_log_labels,
              config):
    print_label(s_train_log_labels)
    print_label(s_test_log_labels)
    print_label(t_pool_log_labels)
    print_label(t_test_log_labels)

    model_path = config['model_path']

    model = LogLSTM(config['d_input'], config['d_hidden'], config['num_layers'])


    epoch = config['epoch']

    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=5e-6)

    batch_size = config['batch_size']

    src_dataset = LogDataset(s_train_log_vectors, s_train_log_labels)
    src_train_loader = DataLoader(src_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)

    # total number of target samples
    totality = len(t_pool_log_vectors)

    trg_unlabeled_dataset = LogDataset(t_pool_log_vectors, t_pool_log_labels)
    trg_unlabeled_loader = DataLoader(trg_unlabeled_dataset, batch_size=batch_size,
                                      shuffle=True, drop_last=False)
    s_best_f1, s_best_pre, s_best_re = 0, 0, 0
    t_best_f1, t_best_pre, t_best_re = 0, 0, 0
    s_best_epoch = 0
    t_best_epoch = 0
    max_epoch = max(config['active_learning']['s_epoch'])

    trg_selected_loader = None

    for epoch in tqdm(range(1, config['epoch'] + 1)):
        model.train()
        iter_per_epoch = max(len(src_train_loader), len(trg_unlabeled_loader))

        for batch_idx in range(iter_per_epoch):
            if batch_idx % len(src_train_loader) == 0:
                src_iter = iter(src_train_loader)
            if batch_idx % len(trg_unlabeled_loader) == 0:
                trg_unlabeled_iter = iter(trg_unlabeled_loader)
            if trg_selected_loader:
                if batch_idx % len(trg_selected_loader) == 0:
                    trg_selected_iter = iter(trg_selected_loader)

            src_data = next(src_iter)
            trg_unlabeled_data = next(trg_unlabeled_iter)

            (src_log_vectors, src_log_labels) = src_data['vec'], src_data['lab']
            src_log_vectors, src_log_labels = src_log_vectors.cuda(), src_log_labels.cuda()

            optimizer.zero_grad()

            total_loss = 0
            # supervised loss on label source data
            src_out = model(src_log_vectors)
            loss1 = criterion(src_out, src_log_labels)

            total_loss += loss1

            # supervised loss on selected target data
            if trg_selected_loader:
                trg_data = next(trg_selected_iter)
                (trg_log_vectors, trg_log_labels) = trg_data['vec'], trg_data['lab']
                trg_log_vectors, trg_log_labels = trg_log_vectors.cuda(), trg_log_labels.cuda()

                trg_selected_out = model(trg_log_vectors)
                selected_nll_loss = criterion(trg_selected_out, trg_log_labels)

                total_loss += selected_nll_loss

            total_loss.backward()
            optimizer.step()

        # test every epoch
        print('\n````````````````````source````````````````````')
        s_f1, s_pre, s_re = testing(s_test_log_vectors, s_test_log_labels, model, config)
        print('````````````````````target````````````````````')
        t_f1, t_pre, t_re = testing(t_test_log_vectors, t_test_log_labels, model, config)
        if epoch >= max_epoch:
            if s_f1 > s_best_f1:
                s_best_f1 = s_f1
                s_best_pre = s_pre
                s_best_re = s_re
                s_best_epoch = epoch
            if t_f1 > t_best_f1:
                t_best_f1 = t_f1
                t_best_pre = t_pre
                t_best_re = t_re
                t_best_epoch = epoch

        # active learning
        if not config['active_learning']['random']:
            model.eval()
            first_stat = list()
            with torch.no_grad():
                for _, tgt_data in enumerate(trg_unlabeled_loader):
                    # tgt_path, tgt_index = data['path'], data['index']
                    tgt_log_vectors, tgt_log_labels, tgt_index = tgt_data['vec'], tgt_data['lab'], tgt_data['index']
                    tgt_log_vectors, tgt_log_labels = tgt_log_vectors.cuda(), tgt_log_labels.cuda()

                    tgt_out = model(tgt_log_vectors)

                    # Uncertainty sampling
                    min2 = torch.topk(tgt_out, k=2, dim=1, largest=False).values
                    mvsm_uncertainty = min2[:, 0] - min2[:, 1]

                    # free energy sampling
                    output_div_t = -1.0 * tgt_out / config['ENERGY_BETA']
                    output_logsumexp = torch.logsumexp(output_div_t, dim=1, keepdim=False)
                    free_energy = -1.0 * config['ENERGY_BETA'] * output_logsumexp

                    for i in range(len(free_energy)):
                        first_stat.append(
                            [tgt_log_vectors[i].cpu().numpy(), tgt_log_labels[i].item(), tgt_index[i].item(),
                             mvsm_uncertainty[i].item(), free_energy[i].item()])

            first_sample_ratio = config['FIRST_SAMPLE_RATIO']
            first_sample_num = math.ceil(totality * first_sample_ratio)
            second_sample_ratio = config['active_learning']['active_ratio'] / first_sample_ratio
            second_sample_num = math.ceil(first_sample_num * second_sample_ratio)


            first_stat = sorted(first_stat, key=lambda x: x[4], reverse=True)
            second_stat = first_stat[:first_sample_num]


            second_stat = sorted(second_stat, key=lambda x: x[3], reverse=True)
            second_stat = second_stat[:second_sample_num]

            active_vec = np.array([item[0] for item in second_stat])
            active_lab = np.array([item[1] for item in second_stat])
            candidate_ds_index = np.array([item[2] for item in second_stat])
            if epoch in config['active_learning']['s_epoch']:
                print('select')
                trg_unlabeled_dataset.remove_item(candidate_ds_index)
                if trg_selected_loader:
                    trg_selected_dataset.add_item(active_vec, active_lab)
                else:
                    trg_selected_dataset = LogDataset(active_vec, active_lab)
                    trg_selected_loader = DataLoader(trg_selected_dataset, batch_size=batch_size,
                                                     shuffle=True, drop_last=False)
            print(f'select samples:{len(active_vec)}')
            print_label(active_lab)
        else:
            if epoch in config['active_learning']['s_epoch']:
                print('select')
                length = len(trg_unlabeled_dataset)
                index = random.sample(range(length), round(totality * config['active_learning']['active_ratio']))
                random_vec = np.array([trg_unlabeled_dataset[i]['vec'] for i in index])
                random_lab = np.array([trg_unlabeled_dataset[i]['lab'] for i in index])
                if trg_selected_loader:
                    trg_selected_dataset.add_item(random_vec, random_lab)
                else:
                    trg_selected_dataset = LogDataset(random_vec, random_lab)
                    trg_selected_loader = DataLoader(trg_selected_dataset, batch_size=batch_size,
                                                     shuffle=True, drop_last=False)

                trg_unlabeled_dataset.remove_item(index)
                print(f'random select samples:{len(random_vec)}')
                print_label(random_lab)
    torch.save(model.state_dict(), model_path)
    print(
        "source: F1: {:.2f}%, Precision: {:.2f}%, Recall: {:.2f}%, epoch:{}".format(
            s_best_f1 * 100, s_best_pre * 100, s_best_re * 100, s_best_epoch
        ))
    print(
        "target: F1: {:.2f}%, Precision: {:.2f}%, Recall: {:.2f}%, epoch:{}".format(
            t_best_f1 * 100, t_best_pre * 100, t_best_re * 100, t_best_epoch
        ))
    return model

def LogAction_without_transfer(s_test_log_vectors, s_test_log_labels,
                               t_pool_log_vectors, t_pool_log_labels, t_test_log_vectors, t_test_log_labels,
                               config):
    print_label(s_test_log_labels)
    print_label(t_pool_log_labels)
    print_label(t_test_log_labels)
    model_path = config['model_path']

    model = LogLSTM(config['d_input'], config['d_hidden'], config['num_layers'])


    epoch = config['epoch']

    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=5e-6)

    batch_size = config['batch_size']


    # total number of target samples
    totality = len(t_pool_log_vectors)

    trg_unlabeled_dataset = LogDataset(t_pool_log_vectors, t_pool_log_labels)
    trg_unlabeled_loader = DataLoader(trg_unlabeled_dataset, batch_size=batch_size,
                                      shuffle=True, drop_last=False)
    s_best_f1, s_best_pre, s_best_re = 0, 0, 0
    t_best_f1, t_best_pre, t_best_re = 0, 0, 0
    s_best_epoch = 0
    t_best_epoch = 0
    max_epoch = max(config['active_learning']['s_epoch'])

    trg_selected_loader = None

    for epoch in tqdm(range(1, config['epoch'] + 1)):
        model.train()
        iter_per_epoch = len(trg_unlabeled_loader)

        for batch_idx in range(iter_per_epoch):

            if trg_selected_loader:
                if batch_idx % len(trg_selected_loader) == 0:
                    trg_selected_iter = iter(trg_selected_loader)
            optimizer.zero_grad()

            total_loss = 0

            # supervised loss on selected target data
            if trg_selected_loader:
                trg_data = next(trg_selected_iter)
                (trg_log_vectors, trg_log_labels) = trg_data['vec'], trg_data['lab']
                trg_log_vectors, trg_log_labels = trg_log_vectors.cuda(), trg_log_labels.cuda()

                trg_selected_out = model(trg_log_vectors)
                selected_nll_loss = criterion(trg_selected_out, trg_log_labels)

                total_loss += selected_nll_loss

                total_loss.backward()
                optimizer.step()

        # test every epoch
        print('\n````````````````````source````````````````````')
        s_f1, s_pre, s_re = testing(s_test_log_vectors, s_test_log_labels, model, config)
        print('````````````````````target````````````````````')
        t_f1, t_pre, t_re = testing(t_test_log_vectors, t_test_log_labels, model, config)
        if s_f1 > s_best_f1:
            s_best_f1 = s_f1
            s_best_pre = s_pre
            s_best_re = s_re
            s_best_epoch = epoch
        if t_f1 > t_best_f1:
            t_best_f1 = t_f1
            t_best_pre = t_pre
            t_best_re = t_re
            t_best_epoch = epoch

        if not config['active_learning']['random']:
            model.eval()
            first_stat = list()
            with torch.no_grad():
                for _, tgt_data in enumerate(trg_unlabeled_loader):
                    tgt_log_vectors, tgt_log_labels, tgt_index = tgt_data['vec'], tgt_data['lab'], tgt_data['index']
                    tgt_log_vectors, tgt_log_labels = tgt_log_vectors.cuda(), tgt_log_labels.cuda()

                    tgt_out = model(tgt_log_vectors)

                    # uncertainty sampling
                    min2 = torch.topk(tgt_out, k=2, dim=1, largest=False).values
                    mvsm_uncertainty = min2[:, 0] - min2[:, 1]

                    # free energy sampling
                    output_div_t = -1.0 * tgt_out / config['ENERGY_BETA']
                    output_logsumexp = torch.logsumexp(output_div_t, dim=1, keepdim=False)
                    free_energy = -1.0 * config['ENERGY_BETA'] * output_logsumexp

                    for i in range(len(free_energy)):
                        first_stat.append(
                            [tgt_log_vectors[i].cpu().numpy(), tgt_log_labels[i].item(), tgt_index[i].item(),
                             mvsm_uncertainty[i].item(), free_energy[i].item()])

            # Here all are selected for the first time because there is no transfer learning.
            first_sample_ratio = 1
            first_sample_num = math.ceil(totality * first_sample_ratio)
            second_sample_ratio = config['active_learning']['active_ratio'] / first_sample_ratio
            second_sample_num = math.ceil(first_sample_num * second_sample_ratio)

            # the first sample using \mathca{F}, higher value, higher consideration
            first_stat = sorted(first_stat, key=lambda x: x[4], reverse=True)
            second_stat = first_stat[:first_sample_num]

            # the second sample using \mathca{U}, higher value, higher consideration
            second_stat = sorted(second_stat, key=lambda x: x[3], reverse=True)
            second_stat = second_stat[:second_sample_num]

            active_vec = np.array([item[0] for item in second_stat])
            active_lab = np.array([item[1] for item in second_stat])
            candidate_ds_index = np.array([item[2] for item in second_stat])
            if epoch in config['active_learning']['s_epoch']:
                print('select')
                trg_unlabeled_dataset.remove_item(candidate_ds_index)
                if trg_selected_loader:
                    trg_selected_dataset.add_item(active_vec, active_lab)
                else:
                    trg_selected_dataset = LogDataset(active_vec, active_lab)
                    trg_selected_loader = DataLoader(trg_selected_dataset, batch_size=batch_size,
                                                     shuffle=True, drop_last=False)
            print(f'select samples:{len(active_vec)}')
            print_label(active_lab)
        else:
            if epoch in config['active_learning']['s_epoch']:
                print('select')
                length = len(trg_unlabeled_dataset)
                index = random.sample(range(length), round(totality * config['active_learning']['active_ratio']))
                random_vec = np.array([trg_unlabeled_dataset[i]['vec'] for i in index])
                random_lab = np.array([trg_unlabeled_dataset[i]['lab'] for i in index])
                if trg_selected_loader:
                    trg_selected_dataset.add_item(random_vec, random_lab)
                else:
                    trg_selected_dataset = LogDataset(random_vec, random_lab)
                    trg_selected_loader = DataLoader(trg_selected_dataset, batch_size=batch_size,
                                                     shuffle=True, drop_last=False)

                trg_unlabeled_dataset.remove_item(index)
                print(f'random select samples:{len(random_vec)}')
                print_label(random_lab)
    torch.save(model.state_dict(), model_path)
    print(
        "source: F1: {:.2f}%, Precision: {:.2f}%, Recall: {:.2f}%, epoch:{}".format(
            s_best_f1 * 100, s_best_pre * 100, s_best_re * 100, s_best_epoch
        ))
    print(
        "target: F1: {:.2f}%, Precision: {:.2f}%, Recall: {:.2f}%, epoch:{}".format(
            t_best_f1 * 100, t_best_pre * 100, t_best_re * 100, t_best_epoch
        ))
    return model



def testing(test_log_vectors, test_log_labels, model, config):
    batch_size = config['batch_size']
    dataset = TensorDataset(torch.FloatTensor(test_log_vectors), torch.tensor(test_log_labels))
    test_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
    all_labels = []
    all_pred_label = []
    model.to(device)
    model.eval()
    with torch.no_grad():
        for step, (vectors, label) in enumerate(test_loader):
            out = model(vectors.to(device))
            # pred_label = torch.argsort(out, 1)[:, 1].cpu().detach().numpy()
            pred_label = F.softmax(out)[:, 1].cpu().detach().numpy()
            all_labels.append(label.cpu().detach().numpy())
            all_pred_label.append(pred_label)
        all_labels = np.concatenate(all_labels)
        all_pred_label = np.concatenate(all_pred_label)
        threshold = config['threshold']
        pred = (all_pred_label > threshold).astype(int)
        tot_labels = np.array(all_labels)
        TP = ((pred == 1).astype(int) + (tot_labels == 1).astype(int) == 2).sum()
        FP = ((pred == 1).astype(int) + (tot_labels == 0).astype(int) == 2).sum()
        FN = ((pred == 0).astype(int) + (tot_labels == 1).astype(int) == 2).sum()
        TN = ((pred == 0).astype(int) + (tot_labels == 0).astype(int) == 2).sum()
        precision = 1.0 * TP / (TP + FP)
        recall = 1.0 * TP / (TP + FN)
        F1 = 2.0 * precision * recall / (precision + recall)

        print(
            "F1: {:.2f}%, Precision: {:.2f}%, Recall: {:.2f}%".format(
                F1 * 100, precision * 100, recall * 100
            ))
        return F1, precision, recall
