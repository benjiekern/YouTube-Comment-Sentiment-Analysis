import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

# Sentiment Model
class SentimentModel(nn.Module):
    def __init__(self, vocab_size, embed_size, lstm_units):
        super(SentimentModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)

        # BiLSTM
        self.lstm = nn.LSTM(
            input_size=embed_size,
            hidden_size=lstm_units,
            num_layers=2,
            batch_first=True,
            bidirectional=True
        )

        # Fully connected layers
        self.dropout = nn.Dropout(0.4)
        self.fc = nn.Linear(lstm_units * 2, 64)
        self.relu = nn.ReLU()
        self.out = nn.Linear(64, 3)

    def forward(self, x):
        x = self.embedding(x)
        lstm_out, (h_n, c_n) = self.lstm(x)

        # Concatenate last hidden states from both directions
        h_forward = h_n[-2, :, :]
        h_backward = h_n[-1, :, :]
        h = torch.cat((h_forward, h_backward), dim=1)
        h = self.dropout(h)
        h = self.fc(h)
        h = torch.relu(h)
        out = self.out(h)
        return out