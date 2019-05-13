import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pdb


class LSTM(nn.Module):
    def __init__(self,
                 vocab_size,
                 embedding_dim,
                 hidden_size,
                 output_size,
                 padding_idx,
                 ckpt=None):
        super(LSTM, self).__init__()
        self._emb = nn.Embedding(vocab_size, embedding_dim, padding_idx=padding_idx)
        self._hidden_size = hidden_size
        self._final_pred = nn.Linear(hidden_size, output_size)

        self._rnn = nn.LSTM(embedding_dim,
                            hidden_size,
                            batch_first=True,
                            bidirectional=False)
        return

    def forward(self, src, src_lengths):
        src_emb = self._emb(src)
        packed_src = nn.utils.rnn.pack_padded_sequence(src_emb,
                                                       src_lengths,
                                                       batch_first=True)
        outputs, (hidden, cell) = self._rnn(packed_src)
        hidden = hidden.permute(1, 0, 2).squeeze()
        pred_scores = self._final_pred(hidden)
        return pred_scores


class RNN(nn.Module):
    def __init__(self,
                 vocab_size,
                 embedding_dim,
                 hidden_size,
                 output_size,
                 ckpt=None):
        super(RNN, self).__init__()
        self._emb = nn.Embedding(vocab_size, embedding_dim)
        self._hidden_size = hidden_size
        self._final_pred = nn.Linear(hidden_size, output_size)

        self._rnn = nn.RNN(embedding_dim,
                           hidden_size,
                           batch_first=True,
                           bidirectional=False)
        return

    def forward(self, src, src_lengths):

        src_emb = self._emb(src)
        packed_src = nn.utils.rnn.pack_padded_sequence(src_emb,
                                                       src_lengths,
                                                       batch_first=True)
        outputs, hidden = self._rnn(packed_src)
        hidden = hidden.permute(1, 0, 2).squeeze()
        pred_scores = self._final_pred(hidden)
        return pred_scores


if __name__ == '__main__':
    model = LSTM(5, 10, 20, 2)
    inputs = torch.LongTensor(np.random.randn(3, 10, 5))
    len_tensor = torch.LongTensor(np.ones((3)) * 10)
    _ = model(inputs, len_tensor)