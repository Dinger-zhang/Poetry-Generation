import torch
import torch.nn as nn
import math
from config import Config

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=500):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        # 初始形状: (max_len, d_model)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # 改为 (max_len, 1, d_model) 以便广播到 (seq_len, batch, d_model)
        pe = pe.unsqueeze(1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: (seq_len, batch, d_model)
        seq_len = x.size(0)
        # 动态扩展 pe（如果需要）
        if seq_len > self.pe.size(0):
            new_pe = torch.zeros(seq_len, 1, self.pe.size(2), device=self.pe.device)
            position = torch.arange(0, seq_len, dtype=torch.float, device=self.pe.device).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, self.pe.size(2), 2).float().to(self.pe.device) * (-math.log(10000.0) / self.pe.size(2)))
            new_pe[:, 0, 0::2] = torch.sin(position * div_term)
            new_pe[:, 0, 1::2] = torch.cos(position * div_term)
            self.pe = new_pe
        x = x + self.pe[:seq_len, :, :]   # (seq_len, 1, d_model) 广播到 (seq_len, batch, d_model)
        return self.dropout(x)

class TransformerPoetryModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_heads=8, num_layers=6, max_len=125, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.pos_encoder = PositionalEncoding(embedding_dim, dropout, max_len=Config.transformer_max_len)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=embedding_dim * 4,
            dropout=dropout,
            batch_first=False
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(embedding_dim, vocab_size)

    def forward(self, x, mask=None):
        seq_len, batch_size = x.size()
        embed = self.embedding(x)          # (seq, batch, embed)
        embed = self.pos_encoder(embed)    # 加位置编码
        if mask is None:
            mask = torch.triu(torch.ones(seq_len, seq_len) * float('-inf'), diagonal=1).to(x.device)
        output = self.transformer(embed, mask=mask)
        logits = self.fc(output)
        return logits, None