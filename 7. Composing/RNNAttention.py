import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):
    def __init__(self, in_channels, hidden_size, seq_len):
        super(Attention, self).__init__()
        self.in_channels = in_channels
        self.hidden_size = hidden_size
        self.seq_len = seq_len
        
        self.lstm = nn.LSTM(in_channels, hidden_size, batch_first=True)
        self.linear = nn.Linear(hidden_size, 1)
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):
        x, _ = self.lstm(x)
        
        e = self.linear(x)
        e = self.tanh(e)
        e = e.view(-1, self.seq_len)
        alpha = self.softmax(e)
        
        c = x.mul(alpha.unsqueeze(-1))
        c = c.sum(axis=1)
        
        return c, alpha
    
class LSTMOutputOnly(nn.Module):
    def __init__(self, in_channels, hidden_size):
        super(LSTMOutputOnly, self).__init__()
        self.lstm = nn.LSTM(in_channels, hidden_size, batch_first=True)
        
    def forward(self, x, state):
        x, _ = self.lstm(x, state)
        return x, None


class RNNAttention(nn.Module):
    def __init__(self, n_notes, n_durations, embed_size=100, rnn_units=256, seq_len=32, use_attention=False):
        super(RNNAttention, self).__init__()
        
        self.embedding1 = nn.Embedding(num_embeddings=n_notes,
                                       embedding_dim=embed_size)
        self.embedding2 = nn.Embedding(num_embeddings=n_durations,
                                       embedding_dim=embed_size)
        
        self.lstm = nn.LSTM(embed_size * 2, rnn_units, batch_first=True)
        
        if use_attention:
            self.layer1 = Attention(rnn_units, rnn_units, seq_len)
        else:
            self.layer1 = LSTMOutputOnly(rnn_units, rnn_units)
            
        self.linear1 = nn.Linear(rnn_units, n_notes)
        self.linear2 = nn.Linear(rnn_units, n_durations)
        
    def forward(self, x):
        embed1 = self.embedding1(x[0])
        embed2 = self.embedding2(x[1])
        # out shape: (batch_size, seq_len, 2 * embed_size)
        out = torch.cat([embed1, embed2], dim=2)
        
        # out shape: (batch_size, seq_len, rnn_units)
        out, _ = self.lstm(out)
        # out shape: (batch_size, rnn_units)
        out, alpha = self.layer1(out)
        
        # notes_out shape: (batch_size, n_notes)
        notes_out = self.linear1(out)
        
        # notes_out shape: (batch_size, n_durations)
        durations_out = self.linear2(out)
            
        return [notes_out, durations_out], alpha
        
        