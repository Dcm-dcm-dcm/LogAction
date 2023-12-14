import copy
import torch.nn as nn

from .Embed import PositionalEncoder
from .Layers import EncoderLayer
from .Sublayers import Norm
from ...anomaly_detection.model.Models import LogLSTM


def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class LogEncoder(nn.Module):
    def __init__(self, d_input, d_output,N, heads, dropout, max_seq_len):
        super().__init__()
        self.d_input = d_input
        self.encoder = Encoder(d_input, d_output, N, heads, dropout, max_seq_len)
        self.decoder1 = nn.Linear(d_output, d_input)
        self.decoder2 = LogLSTM(d_output, 64, 2)

    def forward(self, x, mask=None):
        x = self.encoder(x, mask)
        out1 = self.decoder1(x)
        out2 = self.decoder2(x)
        return out1,out2

class Encoder(nn.Module):
    def __init__(self, d_input, d_model, N, heads, dropout, max_seq_len):
        super().__init__()
        self.N = N
        self.embed = nn.Linear(d_input, d_model)
        self.pe = PositionalEncoder(d_model, max_seq_len=max_seq_len, dropout=dropout)
        self.layers = get_clones(EncoderLayer(d_model, heads, dropout), N)
        self.norm = Norm(d_model)

    def forward(self, src, mask=None):
        x = self.embed(src.float())
        x = self.pe(x)
        for i in range(self.N):
            x = self.layers[i](x, mask=mask)
        return self.norm(x)


