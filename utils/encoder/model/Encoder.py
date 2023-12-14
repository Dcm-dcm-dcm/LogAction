import torch.nn as nn
from ...anomaly_detection.model.Models import LogLSTM



class LSTMEncoder(nn.Module):

    def __init__(self, emb_dim, hid_dim, output_dim):
        super().__init__()
        self.emb_dim = emb_dim
        self.hid_dim = hid_dim
        self.output_dim = output_dim
        self.rnn = nn.LSTM(emb_dim, hid_dim, 2,  bidirectional=False, batch_first=True)

        self.discriminator = LogLSTM(hid_dim, 64, 2)

    def forward(self, input, alpha=0.1):
        output, (hidden, cell) = self.rnn(input)
        y1 = self.discriminator(output)
        return output, y1