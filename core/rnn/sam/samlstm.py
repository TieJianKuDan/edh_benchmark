import random

import torch
import torch.nn as nn


class SAM(nn.Module):
    '''
    self_attention_memory_module
    '''
    def __init__(self, input_channel, hidden_channel):
        super(SAM, self).__init__()
        # h(hidden): layer q, k, v
        # m(memory): layer k2, v2
        # layer z, m are for layer after concat(attention_h, attention_m)

        # layer_q, k, v are for h (hidden) layer
        # Layer_k2, v2 are for m (memory) layer
        # Layer_z, m are using after concatinating attention_h and attention_m layer

        self.layer_q = nn.Conv2d(input_channel, hidden_channel, 1)
        self.layer_k = nn.Conv2d(input_channel, hidden_channel, 1)
        self.layer_k2 = nn.Conv2d(input_channel, hidden_channel, 1)
        self.layer_v = nn.Conv2d(input_channel, input_channel, 1)
        self.layer_v2 = nn.Conv2d(input_channel, input_channel, 1)
        self.layer_z = nn.Conv2d(input_channel*2, input_channel*2, 1)
        self.layer_m = nn.Conv2d(input_channel*3, input_channel*3, 1)
        self.hidden_channel = hidden_channel
        self.input_channel = input_channel

    def forward(self, h, m):
        batch_size, channel, H, W = h.shape
        # feature aggregation
        ##### hidden h attention #####
        K_h = self.layer_k(h)
        Q_h = self.layer_q(h)
        K_h = K_h.view(batch_size, self.hidden_channel, H * W)
        Q_h = Q_h.view(batch_size, self.hidden_channel, H * W)
        Q_h = Q_h.transpose(1, 2)

        A_h = torch.softmax(torch.bmm(Q_h, K_h), dim=-1)  # batch_size, H*W, H*W

        V_h = self.layer_v(h)
        V_h = V_h.view(batch_size, self.input_channel, H * W)
        Z_h = torch.matmul(A_h, V_h.permute(0, 2, 1))

        ###### memory m attention #####
        K_m = self.layer_k2(m)
        V_m = self.layer_v2(m)
        K_m = K_m.view(batch_size, self.hidden_channel, H * W)
        V_m = V_m.view(batch_size, self.input_channel, H * W)
        A_m = torch.softmax(torch.bmm(Q_h, K_m), dim=-1)
        V_m = self.layer_v2(m)
        V_m = V_m.view(batch_size, self.input_channel, H * W)
        Z_m = torch.matmul(A_m, V_m.permute(0, 2, 1))
        Z_h = Z_h.transpose(1, 2).view(batch_size, self.input_channel, H, W)
        Z_m = Z_m.transpose(1, 2).view(batch_size, self.input_channel, H, W)

        ### Z_h & Z_m (from attention) then, concat then computation ####
        W_z = torch.cat([Z_h, Z_m], dim=1)
        Z = self.layer_z(W_z)
        ## Memory Updating (Ref: SA-ConvLSTM)
        combined = self.layer_m(torch.cat([Z, h], dim=1))  # 3 * input_channel
        mo, mg, mi = torch.split(combined, self.input_channel, dim=1)
        ### (Ref: SA-ConvLSTM)
        mi = torch.sigmoid(mi)
        new_m = (1 - mi) * m + mi * torch.tanh(mg)
        new_h = torch.sigmoid(mo) * new_m

        return new_h, new_m


class SAMConvLSTMCell(nn.Module):

    def __init__(self, config):
        super(SAMConvLSTMCell, self).__init__()
        self.hidden_channel = config['hidden_channel']
        self.attention_layer = SAM(
            config['hidden_channel'], config['att_hidden_dim']
        ) # 32, 16
        padding = config['kernel_size'][0] // 2, config['kernel_size'][1] // 2
        self.conv2d = nn.Sequential(
            nn.Conv2d(
                in_channels=config['hidden_channel'] + config['hidden_channel'], 
                out_channels=4*config['hidden_channel'],
                kernel_size=config['kernel_size'], padding=padding
            ), 
            nn.GroupNorm(4*config['hidden_channel'], 4*config['hidden_channel'])
        )  # (num_groups, num_channels)


    def forward(self, x, hidden):
        h, c, m = hidden
        
        combined = torch.cat([x, h], dim=1)
        
        combined_conv = self.conv2d(combined)
        i, f, o, g = torch.split(combined_conv, self.hidden_channel, dim=1)
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        o = torch.sigmoid(o)
        g = torch.tanh(g)
        c_next = f * c + i * g
        h_next = o * torch.tanh(c_next)

        h_next, m_next = self.attention_layer(h_next, m)

        return h_next, (h_next, c_next, m_next)

    def init_hidden(self, batch_size, hidden_size, device):
        h, w = hidden_size
        
        return (
            torch.zeros(
                batch_size, self.hidden_channel, h, w
            ).to(device),
            torch.zeros(
                batch_size, self.hidden_channel, h, w
            ).to(device),
            torch.zeros(
                batch_size, self.hidden_channel, h, w
            ).to(device)
        )
    

class SAMConvLSTM(nn.Module):
    '''
    self-attention convlstm
    '''
    def __init__(self, config):
        super(SAMConvLSTM, self).__init__()
        self.cond_len = config['cond_len']
        self.pred_len = config['pred_len']
        self.n_layers = config['n_layers']
        self.hidden_size = config['hidden_size']
        self.cells, self.bns = [], []

        for _ in range(config['n_layers']):
            self.cells.append(SAMConvLSTMCell(config))
            self.bns.append(
                nn.LayerNorm(
                    (config['hidden_channel'], 16, 16)
                )
            )  # Use layernorm

        self.cells = nn.ModuleList(self.cells)
        self.bns = nn.ModuleList(self.bns)

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

        # Prediction layer
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
            out_channels=1,
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

        for t in range(self.cond_len + self.pred_len - 1):
            if t < self.cond_len \
                or random.random() < teacher_forcing_rate:
                x = frames[:, t, :, :, :]
            else:
                x = out

            x = self.img_encode(x)

            for i, cell in enumerate(self.cells):
                out, hidden[i] = cell(x, hidden[i])
                out = self.bns[i](out)

            # out = self.decoder_predict(out)
            out = self.img_decode(out)
            preds.append(out)

        preds = torch.stack(preds, dim=1)

        return preds


    def init_hidden(self, batch_size, hidden_size, device):
        states = []
        for i in range(self.n_layers):
            states.append(
                self.cells[i].init_hidden(
                    batch_size, hidden_size, device
                )
            )

        return states