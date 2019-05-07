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
                 ckpt=None):
        super(LSTM, self).__init__()
        self.emb = nn.Embedding(vocab_size, embedding_dim)
        self.hidden_size = hidden_size
        self.output_size = output_size
        # self.out_pred = nn.Linear(hidden_size, output_size)
        self.rnn = nn.GRU(
            embedding_dim,
            self.hidden_size,
            batch_first=True,
            bidirectional=False)
        self.seq2pred = nn.Linear(self.hidden_size, self.output_size)

        self._ckpt = ckpt
        return

    def forward(self, src, src_lengths):

        src_emb = self.emb(src)
        packed_src = nn.utils.rnn.pack_padded_sequence(
            src_emb, src_lengths, batch_first=True)
        packed_output, hidden = self.rnn(packed_src)
        pdb.set_trace()
        output, _ = nn.utils.rnn.pad_packed_sequence(
            packed_output, batch_first=True)
        pdb.set_trace()
        pred = self.seq2pred(output)
        # pdb.set_trace()
        pred_scores = F.log_softmax(pred, dim=2)

        # fwd_final = hidden[0:hidden.size(0):2]
        # bwd_final = hidden[1:hidden.size(0):2]
        # final = torch.cat([fwd_final, bwd_final], dim=2)
        return pred_scores, hidden

if __name__ == '__main__':
    model = LSTM(5, 10, 20, 2)
    inputs = torch.LongTensor(np.random.randn(3, 10, 5))
    len_tensor = torch.LongTensor(np.ones((3))*10)
    _ = model(inputs, len_tensor)