import torch
import torch.nn as nn
from .Attention import LinearAttention
from torch.nn.parameter import Parameter

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class LogLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(LogLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.atten_guide = Parameter(torch.Tensor(self.hidden_size))
        self.atten_guide.data.normal_(0, 1)
        self.atten = LinearAttention(tensor_1_dim=self.hidden_size, tensor_2_dim=self.hidden_size)

        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.Linear(64, 2)
        )

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        batch_size = x.size(0)
        atten_guide = torch.unsqueeze(self.atten_guide, dim=1).expand(-1, batch_size)
        atten_guide = atten_guide.transpose(1, 0)

        hiddens, _ = self.lstm(x, (h0, c0))

        sent_probs = self.atten(atten_guide, hiddens)
        batch_size, srclen, dim = hiddens.size()
        sent_probs = sent_probs.view(batch_size, srclen, -1)
        represents = hiddens * sent_probs
        represents = represents.sum(dim=1)

        out = self.fc(represents)

        return out

