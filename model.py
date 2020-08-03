from main import *
from config import *
import torch.nn as nn
import torch as t


class PoetryModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super(PoetryModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=Config.num_layers)
        self.linear = nn.Linear(self.hidden_dim, vocab_size)

    def forward(self, input, hidden: dict = None):
        device = Config.device
        seq_len, batch_size = input.size()
        if hidden is None:
            h_0 = t.zeros(Config.num_layers, batch_size, self.hidden_dim)
            c_0 = t.zeros(Config.num_layers, batch_size, self.hidden_dim)
        else:
            h_0, c_0 = hidden
        embeds = self.embeddings(input)
        output, hidden = self.lstm(embeds, (h_0, c_0))
        output = self.linear(output.view(seq_len*batch_size, -1))
        return output, hidden
