import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class LSTM(nn.Module):
    def __init__(self,
                 vocab_size,
                 embedding_dim,
                 hidden_size,
                 output_size,
                #  use_gpu=False,
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
        # print('->', src.device)
        src_emb = self.emb(src)
        packed_src = nn.utils.rnn.pack_padded_sequence(
            src_emb, src_lengths.cpu().numpy(), batch_first=True)
        packed_output, hidden = self.rnn(packed_src)
        output, _ = nn.utils.rnn.pad_packed_sequence(
            packed_output, batch_first=True)
        pred = self.seq2pred(output)
        # pdb.set_trace()
        pred_scores = F.log_softmax(pred, dim=2)

        # fwd_final = hidden[0:hidden.size(0):2]
        # bwd_final = hidden[1:hidden.size(0):2]
        # final = torch.cat([fwd_final, bwd_final], dim=2)
        return pred_scores, hidden

    def restore(self):
        self.load_state_dict(torch.load(self._ckpt))
        return