import random

import torch
import torch.nn as nn


class ConvLSTMCell(nn.Module):

    def __init__(self, in_channels, h_channels, kernel_size, bias=True):
        """
        Initialize ConvLSTM cell.
        Parameters
        ----------
        in_channel: int
            Number of channels of input tensor.
        hidden_channel: int
            Number of channels of hidden state.
        kernel_size: (int, int)
            Size of the convolutional kernel.
        bias: bool
            Whether or not to add the bias.
        """

        super(ConvLSTMCell, self).__init__()
        self.h_channels = h_channels
        padding = kernel_size[0] // 2, kernel_size[1] // 2

        self.conv = nn.Conv2d(
            in_channels=in_channels + h_channels,
            out_channels=4 * h_channels,
            kernel_size=kernel_size,
            padding=padding,
            bias=bias
        )
        
    def forward(self, input_tensor, cur_state):

        h_cur, c_cur = cur_state
        
        combined = torch.cat([input_tensor, h_cur], dim=1)  # concatenate along channel axis

        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(
            combined_conv, self.h_channels, dim=1
        )
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next

    def init_hidden(self, batch_size, hidden_size, device):
        height, width = hidden_size
        
        return (
            torch.zeros(
                batch_size, self.h_channels, height, width
            ).to(device),
            torch.zeros(
                batch_size, self.h_channels, height, width
            ).to(device)
        )


class ConvLSTM(nn.Module):
    def __init__(self, config):
        super(ConvLSTM, self).__init__()
        self.cond_len = config['cond_len']
        self.pred_len = config['pred_len']
        self.n_layers = config['n_layers']
        self.hidden_size = config['hidden_size']
        self.cells = []

        for _ in range(self.n_layers):
            self.cells.append(
                ConvLSTMCell(
                    in_channels=config['hidden_channel'],
                    h_channels=config['hidden_channel'],
                    kernel_size=config['kernel_size'],
                    bias=True
                )
            )
            # self.bns.append(nn.LayerNorm((config['hidden_channel'], 16, 16)))  # Use layernorm
        self.cells = nn.ModuleList(self.cells)
        
        self.img_encode = nn.Sequential(
            nn.Conv2d(
                in_channels=config['in_channel'],
                out_channels=config['hidden_channel'],
                kernel_size=1, stride=1, padding=0,
            ),
            nn.LeakyReLU(0.1),
            nn.Conv2d(
                in_channels=config['hidden_channel'],
                out_channels=config['hidden_channel'],
                kernel_size=3, stride=2, padding=1,
            ),
            nn.LeakyReLU(0.1),
            nn.Conv2d(
                in_channels=config['hidden_channel'],
                out_channels=config['hidden_channel'],
                kernel_size=3, stride=2, padding=1,
            ),
            nn.LeakyReLU(0.1)
        )

        self.img_decode = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=config['hidden_channel'],
                out_channels=config['hidden_channel'],
                kernel_size=3, stride=2, padding=1, output_padding=1,
            ),
            nn.LeakyReLU(0.1),
            nn.ConvTranspose2d(
                in_channels=config['hidden_channel'],
                out_channels=config['hidden_channel'],
                kernel_size=3, stride=2, padding=1, output_padding=1,
            ),
            nn.LeakyReLU(0.1),
            nn.Conv2d(
                in_channels=config['hidden_channel'],
                out_channels=config['in_channel'],
                kernel_size=1, stride=1, padding=0,
            )
        )

        # Linear
        self.decoder_predict = nn.Conv2d(
            in_channels=config['hidden_channel'],
            out_channels=config['hidden_channel'],
            kernel_size=(1, 1),
            padding=(0, 0)
        )

    def forward(
        self, frames, 
        teacher_forcing_rate=0.5, 
        hidden=None
    ):
        '''
        frames shape: (b t c h w)
        '''
        # Init hidden weight
        if hidden == None:
            hidden = self.init_hidden(
                batch_size=frames.shape[0], 
                hidden_size=self.hidden_size,
                device=frames.device
            )

        preds = []

        # Seq2seq
        for t in range(self.cond_len + self.pred_len - 1):
            if t < self.cond_len \
                or random.random() < teacher_forcing_rate:
                x = frames[:, t, :, :, :]
            else:
                x = out
            
            x = self.img_encode(x)

            for i, cell in enumerate(self.cells):
                # hid = cell(input_tensor=x, cur_state=[hid[0], hid[1]])
                hidden[i] = cell(
                    input_tensor=x, cur_state=hidden[i]
                )
                out = self.decoder_predict(hidden[i][0])

            out = self.img_decode(out)
            preds.append(out)

        preds = torch.stack(preds, dim=1)

        return preds


    def init_hidden(self, batch_size, hidden_size, device):
        states = []
        for i in range(self.n_layers):
            states.append(
                self.cells[i].init_hidden(
                    batch_size, hidden_size, device)
            )

        return states
