import torch
import torch.nn as nn


class Prenet(nn.Module):
    def __init__(self, input_dim, dropout_rate=0.5, hidden_dims=[256,128]):
        super(Prenet, self).__init__()
        self.hidden_layer_1 = nn.Linear(input_dim, hidden_dims[0])
        self.hidden_layer_2 = nn.Linear(hidden_dims[0], hidden_dims[1])
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, input_x):
        h1 = self.dropout(self.relu(self.hidden_layer_1(input_x)))
        output = self.dropout(self.relu(self.hidden_layer_2(h1)))
        return output


class Highway(nn.Module):
    def __init__(self, in_size, out_size):
        super(Highway, self).__init__()
        self.H = nn.Linear(in_size, out_size)
        self.H.bias.data.zero_()
        self.T = nn.Linear(in_size, out_size)
        self.T.bias.data.fill_(-1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, inputs):
        H = self.relu(self.H(inputs))
        T = self.sigmoid(self.T(inputs))
        return H * T + inputs * (1.0 - T)


class MLP(nn.Module):
    """ Multi-layer Perceptron: Layers of FC(+act/norm) + linear output """

    def __init__(self, input_size, output_size, hidden, activation='ReLU', dropout_rate=None, bn=False):
        """
        Args:
            input_size: int
            output_size: int
            hidden: str of integers, ex: '256_512'
        """
        super(MLP, self).__init__()
        dims = list(map(int, hidden.split('_')))
        in_size = [input_size] + dims[:-1]
        out_size = dims
        blocks = []
        for in_s, out_s in zip(in_size, out_size):
            blocks.append(FCNorm(in_s, out_s, activation, dropout_rate, bn))
        blocks.append(nn.Linear(dims[-1], output_size))
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x) :
        """
        Inputs:
            x: (batch, timesteps, feature_size)

        Outputs:
            y: (batch, timesteps, symbol_size)
        """
        y = x
        for block in self.blocks:
            y = block(y)
        return y


class FCNorm(nn.Module):
    """ Fully connected [+activation] [+batchnorm] [+dropout] """

    def __init__(self, input_size, output_size, activation='ReLU', dropout_rate=None, bn=False, bias=True):
        super(FCNorm, self).__init__()
        self.m = nn.Linear(input_size, output_size, bias)
        self.act = getattr(nn, activation)() if activation else None
        self.batchnorm = nn.BatchNorm1d(output_size) if bn else None
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate is not None else None

    def forward(self, x):
        """
        Inputs:
            x: (batch, timesteps, input_size)

        Outputs:
            y: (batch, timesteps, output_size)
        """
        h = self.m(x)
        if self.act:
            h = self.act(h)
        if self.batchnorm :
            h = h.permute(0, 2, 1)
            h = self.batchnorm(h)
            h = h.permute(0, 2, 1)
        if self.dropout:
            h = self.dropout(h)
        return h


class Conv1dNorm(nn.Module):
    """ Conv1d + activation + batchnorm """

    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1,
                 padding=None, dilation=1, bias=True, w_init_gain='linear', activation='ReLU'):
        super(Conv1dNorm, self).__init__()
        if padding is None:
            assert(kernel_size % 2 == 1)
            padding = int(dilation * (kernel_size - 1) / 2)

        self.conv = torch.nn.Conv1d(in_channels, out_channels,
                                    kernel_size=kernel_size, stride=stride,
                                    padding=padding, dilation=dilation,
                                    bias=bias)

        nn.init.xavier_uniform_(
            self.conv.weight, gain=nn.init.calculate_gain(w_init_gain))

        self.bn = nn.BatchNorm1d(out_channels, momentum=0.99, eps=1e-3)
        self.activation = getattr(nn, activation)()

    def forward(self, x, batch_lengths=None):
        """
        Inputs:
            x: (batch, in_channel, length)
        Outpurs:
            y: (batch, out_channel, lengths)
        """
        y = self.conv(x)
        if batch_lengths is not None :
            y = masked_out(y, batch_lengths)
        y = self.activation(y)
        y = self.bn(y)
        return y


def masked_out(outputs, lengths, axis=2):
    mask = torch.zeros_like(outputs)
    if axis == 2:
        for i, len_ in enumerate(lengths):
            mask[i, :, :int(len_)] = 1
    elif axis == 1:
        for i, len_ in enumerate(lengths):
            mask[i, :int(len_)] = 1
    else:
        raise('Not implemented: masked_out() with axis not in {1, 2}')
    return outputs * mask


class ResBlock(nn.Module):
    def __init__(self, in_channel=256, out_channel=256, kernel_size=5, activation=nn.ReLU()):
        super(ResBlock, self).__init__()
        self.cn1 = Conv1dNorm(in_channel, out_channel, kernel_size, stride=1, padding=None)
        self.cn2 = Conv1dNorm(out_channel, out_channel, 1, stride=1, padding=None)
    def forward(self, x, lengths=None):
        y1 = self.cn1(x, lengths)
        y2 = self.cn2(y1, lengths)
        return y1 + y2
