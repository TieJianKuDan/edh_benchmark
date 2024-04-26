from einops import rearrange
import torch
from torch import nn
from torch.nn import LSTMCell

class NStepLSTM(nn.Module):
    # predict n step
    def __init__(self, input_channel, hidden_channels, output_channel, pred_len, p=0):
        super(NStepLSTM, self).__init__()
        self.pred_len = pred_len
        self.input_channel = input_channel
        self.hidden_channels = hidden_channels

        lstm = [None] * len(hidden_channels)
        for i in range(len(lstm)):
            if i == 0:
                lstm[i] = LSTMCell(
                    input_channel, hidden_channels[i]
                )
            else:
                lstm[i] = LSTMCell(
                    hidden_channels[i-1], hidden_channels[i]
                )
        self.lstm = nn.Sequential(*lstm)

        self.dropout = nn.Dropout(p=p)
        self.lout = nn.Linear(hidden_channels[-1], output_channel)

    def forward(self, batch):
        # (b, t, f)
        cond_len = batch.shape[1]

        preds = [None] * (cond_len + self.pred_len - 1)

        c = [None] * len(self.lstm)
        h = [None] * len(self.lstm)
        for i in range(len(self.lstm)):
            c[i] = torch.zeros(
                (batch.shape[0], self.hidden_channels[i])
            ).to(batch.device)
            h[i] = torch.zeros(
                (batch.shape[0], self.hidden_channels[i])
            ).to(batch.device)

        for i in range(len(preds)):
            if i < cond_len:
                x = batch[:, i]
            else:
                x = preds[i - 1]
            
            for j in range(len(self.lstm)):
                h[j], c[j] = self.lstm[j](x, (h[j], c[j]))
                h[j], c[j] = self.dropout(h[j]), self.dropout(c[j])
                x = h[j]
            
            preds[i] = h[-1]
            preds[i] = self.lout(preds[i])

        preds = torch.stack(preds, dim=1)
        return preds
