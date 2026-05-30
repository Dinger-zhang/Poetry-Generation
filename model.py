import math

import torch
import torch.nn as nn

from config import Config


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=512):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, 1, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[: x.size(0)]
        return self.dropout(x)


class LSTMPoetryModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super(LSTMPoetryModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, self.hidden_dim, num_layers=Config.num_layers)
        self.linear = nn.Linear(self.hidden_dim, vocab_size)

    def forward(self, input, hidden=None):
        seq_len, batch_size = input.size()
        if hidden is None:
            h_0 = input.data.new(Config.num_layers, batch_size, self.hidden_dim).fill_(0).float()
            c_0 = input.data.new(Config.num_layers, batch_size, self.hidden_dim).fill_(0).float()
        else:
            h_0, c_0 = hidden

        embeds = self.embeddings(input)
        output, hidden = self.lstm(embeds, (h_0, c_0))
        output = self.linear(output.view(seq_len * batch_size, -1))
        return output, hidden


class TransformerPoetryModel(nn.Module):
    def __init__(
        self,
        vocab_size,
        embedding_dim=Config.embedding_dim,
        hidden_dim=None,
        nhead=Config.transformer_nhead,
        num_layers=Config.transformer_num_layers,
        dim_feedforward=Config.transformer_dim_feedforward,
        dropout=Config.transformer_dropout,
        max_len=Config.max_position_len,
    ):
        super(TransformerPoetryModel, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.max_len = max_len

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.pos_encoder = PositionalEncoding(embedding_dim, dropout, max_len)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(embedding_dim, vocab_size)

    def _generate_square_subsequent_mask(self, seq_len, device):
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
        return mask.masked_fill(mask == 1, float("-inf"))

    def forward(self, input, hidden=None):
        if input.dim() == 1:
            input = input.view(-1, 1)

        input_len = input.size(0)
        tokens = input if hidden is None else torch.cat([hidden, input], dim=0)
        if tokens.size(0) > self.max_len:
            tokens = tokens[-self.max_len :]

        seq_len, batch_size = tokens.size()
        mask = self._generate_square_subsequent_mask(seq_len, tokens.device)

        embeds = self.embedding(tokens) * math.sqrt(self.embedding_dim)
        embeds = self.pos_encoder(embeds)
        output = self.transformer(embeds, mask=mask)
        output = self.fc(output)

        if hidden is not None:
            output = output[-input_len:]

        output = output.contiguous().view(-1, self.vocab_size)
        next_hidden = tokens.detach() if batch_size == 1 else None
        return output, next_hidden


class PoetryModel(TransformerPoetryModel):
    pass
