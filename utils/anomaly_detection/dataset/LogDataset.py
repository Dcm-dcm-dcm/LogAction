import torch
from torch.utils.data import Dataset
import numpy as np

class LogDataset(Dataset):
    def __init__(self, log_vectors, log_labels):
        self.log_vectors = log_vectors
        self.log_labels = log_labels

    def __len__(self):
        return len(self.log_vectors)

    def __getitem__(self, index):
        samples = {
            'index':index,
            'vec':torch.FloatTensor(self.log_vectors[index]),
            'lab':torch.tensor(self.log_labels[index])
        }
        return samples

    def add_item(self, log_vector_add, log_label_add):

        self.log_vectors = np.concatenate((self.log_vectors, log_vector_add), axis=0)
        self.log_labels = np.concatenate((self.log_labels, log_label_add), axis=0)
        return self.log_vectors,self.log_labels

    def remove_item(self, reduced):
        self.log_vectors = np.delete(self.log_vectors, reduced, axis=0)
        self.log_labels = np.delete(self.log_labels, reduced, axis=0)
        return self.log_vectors,self.log_labels

