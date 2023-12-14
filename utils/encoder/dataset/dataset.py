from torch.utils.data import Dataset
import random
import numpy as np
import torch

def get_mask(log_seqs):
    mask = (log_seqs != torch.zeros_like(log_seqs[0]))
    dim0, dim1, dim2 = mask.shape
    temp0 = []
    for i in range(dim0):
        temp1 = []
        for j in range(dim1):
            temp = False
            for k in range(dim2):
                temp |= mask[i][j][k]
            temp1.append(temp)
        temp0.append(temp1)
    mask = torch.BoolTensor(temp0).unsqueeze(-2)
    return mask

def collate_fn(batch):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mask_seq_batch = torch.as_tensor(np.array([mask_emb for idx, mask_emb, true_emb, mask in batch])).float().to(device)
    true_seq_batch = torch.as_tensor(np.array([true_emb for idx, mask_emb, true_emb, mask in batch])).float().to(device)
    idx_batch = torch.as_tensor(np.array([idx for idx, mask_emb, true_emb, mask in batch]), dtype=torch.long).to(device)
    mask_batch = torch.as_tensor(np.array([mask for idx, mask_emb, true_emb, mask in batch]), dtype=torch.long).to(device)
    return idx_batch, mask_seq_batch, true_seq_batch, mask_batch

def collate_fn_label(batch):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mask_seq_batch = torch.as_tensor(np.array([mask_emb for idx, mask_emb, true_emb, label, mask in batch])).float().to(device)
    true_seq_batch = torch.as_tensor(np.array([true_emb for idx, mask_emb, true_emb, label, mask in batch])).float().to(device)
    label_batch = torch.as_tensor(np.array([label for idx, mask_emb, true_emb, label, mask in batch]), dtype=torch.long).to(device)
    idx_batch = torch.as_tensor(np.array([idx for idx, mask_emb, true_emb, label, mask in batch]), dtype=torch.long).to(device)
    mask_batch = torch.as_tensor(np.array([mask for idx, mask_emb, true_emb, label, mask in batch]), dtype=torch.long).to(device)
    return idx_batch, mask_seq_batch, true_seq_batch, label_batch,mask_batch


class LogEncodingWithLabelDataset(Dataset):
    def __init__(self, log_seqs, log_labels):
        self.log_seqs = log_seqs
        self.log_labels = log_labels

    def __len__(self):
        return len(self.log_seqs)

    def __getitem__(self, item):
        mask_log_seqs,mask = self.random_log_seq(self.log_seqs[item])
        return item, mask_log_seqs,self.log_seqs[item],self.log_labels[item],mask

    def random_log_seq(self, log_seq):
        mask_log_seqs = []
        mask = []
        for log_emb_key in log_seq:
            prob = random.random()
            if prob < 0.15:
                prob /= 0.15

                # 80% randomly change token to mask token
                if prob < 0.8:
                    mask_log_seqs.append(np.zeros_like(log_emb_key))
                    mask.append(False)

                # 10% randomly change token to random token
                elif prob < 0.9:
                    i = np.random.choice(len(self.log_seqs))
                    j = np.random.choice(len(self.log_seqs[i]))
                    mask_log_seqs.append(self.log_seqs[i][j])
                    mask.append(True)
                # 10% randomly change token to current token
                else:
                    mask_log_seqs.append(log_emb_key)
                    mask.append(True)
            else:
                mask_log_seqs.append(log_emb_key)
                mask.append(True)

        return np.array(mask_log_seqs),np.array([np.array(mask)])

