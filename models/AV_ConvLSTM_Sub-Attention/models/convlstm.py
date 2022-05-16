import math
import torch
import torch.nn as nn


def init_layer(layer):
    """Initialize a Linear or Convolutional layer.
    Ref: He, Kaiming, et al. "Delving deep into rectifiers: Surpassing
    human-level performance on imagenet classification." Proceedings of the
    IEEE international conference on computer vision. 2015.

    Input
        layer: torch.Tensor - The current layer of the neural network
    """

    if layer.weight.ndimension() == 4:
        (n_out, n_in, height, width) = layer.weight.size()
        n = n_in * height * width
    elif layer.weight.ndimension() == 3:
        (n_out, n_in, height) = layer.weight.size()
        n = n_in * height
    elif layer.weight.ndimension() == 2:
        (n_out, n) = layer.weight.size()

    std = math.sqrt(2. / n)
    scale = std * math.sqrt(3.)
    layer.weight.data.uniform_(-scale, scale)

    if layer.bias is not None:
        layer.bias.data.fill_(0.)


def init_lstm(layer):
    """
    Initialises the hidden layers in the LSTM - H0 and C0.

    Input
        layer: torch.Tensor - The LSTM layer
    """
    n_i1, n_i2 = layer.weight_ih_l0.size()
    n_i = n_i1 * n_i2

    std = math.sqrt(2. / n_i)
    scale = std * math.sqrt(3.)
    layer.weight_ih_l0.data.uniform_(-scale, scale)

    if layer.bias_ih_l0 is not None:
        layer.bias_ih_l0.data.fill_(0.)

    n_h1, n_h2 = layer.weight_hh_l0.size()
    n_h = n_h1 * n_h2

    std = math.sqrt(2. / n_h)
    scale = std * math.sqrt(3.)
    layer.weight_hh_l0.data.uniform_(-scale, scale)

    if layer.bias_hh_l0 is not None:
        layer.bias_hh_l0.data.fill_(0.)


def init_att_layer(layer):
    """
    Initilise the weights and bias of the attention layer to 1 and 0
    respectively. This is because the first iteration through the attention
    mechanism should weight each time step equally.

    Input
        layer: torch.Tensor - The current layer of the neural network
    """
    layer.weight.data.fill_(1.)

    if layer.bias is not None:
        layer.bias.data.fill_(0.)


def init_bn(bn):
    """
    Initialize a Batchnorm layer.

    Input
        bn: torch.Tensor - The batch normalisation layer
    """

    bn.bias.data.fill_(0.)
    bn.weight.data.fill_(1.)


class ConvBlock1d(nn.Module):
    """
    Creates an instance of a 1D convolutional layer. This includes the
    convolutional filter but also the type of normalisation "batch" or
    "weight", the activation function, and initialises the weights.
    """
    def __init__(self, in_channels, out_channels, kernel, stride, pad,
                 normalisation, dil=1):
        super(ConvBlock1d, self).__init__()
        self.norm = normalisation
        self.conv1 = nn.Conv1d(in_channels=in_channels,
                               out_channels=out_channels,
                               kernel_size=kernel,
                               stride=stride,
                               padding=pad,
                               dilation=dil)
        if self.norm == 'bn':
            self.bn1 = nn.BatchNorm1d(out_channels)
        elif self.norm == 'wn':
            self.conv1 = nn.utils.weight_norm(self.conv1, name='weight')
        else:
            self.conv1 = self.conv1
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        """
        Initialises the weights of the current layer
        """
        init_layer(self.conv1)
        init_bn(self.bn1)

    def forward(self, input):
        """
        Passes the input through the convolutional filter

        Input
            input: torch.Tensor - The current input at this stage of the network
        """
        x = input
        if self.norm == 'bn':
            x = self.relu(self.bn1(self.conv1(x)))
        else:
            x = self.relu(self.conv1(x))

        return x


class ConvBlock2d(nn.Module):
    """
    Creates an instance of a 2D convolutional layer. This includes the
    convolutional filter but also the type of normalisation "batch" or
    "weight", the activation function, and initialises the weights.
    """
    def __init__(self, in_channels, out_channels, kernel, stride, pad,
                 normalisation, att=None):
        super(ConvBlock2d, self).__init__()
        self.norm = normalisation
        self.conv1 = nn.Conv2d(in_channels=in_channels,
                               out_channels=out_channels,
                               kernel_size=kernel,
                               stride=stride,
                               padding=pad)
        if self.norm == 'bn':
            self.bn1 = nn.BatchNorm2d(out_channels)
        elif self.norm == 'wn':
            self.conv1 = nn.utils.weight_norm(self.conv1, name='weight')
        else:
            self.conv1 = self.conv1
        self.att = att
        if not self.att:
            self.act = nn.ReLU()
        else:
            self.norm = None
            if self.att == 'softmax':
                self.act = nn.Softmax(dim=-1)
            elif self.att == 'global':
                self.act = None
            else:
                self.act = nn.Sigmoid()
        self.init_weights()

    def init_weights(self):
        """
        Initialises the weights of the current layer
        """
        if self.att:
            init_att_layer(self.conv1)
        else:
            init_layer(self.conv1)
        init_bn(self.bn1)

    def forward(self, input):
        """
        Passes the input through the convolutional filter

        Input
            input: torch.Tensor - The current input at this stage of the network
        """
        x = input
        if self.att:
            x = self.conv1(x)
            if self.act():
                x = self.act(x)
        else:
            if self.norm == 'bn':
                x = self.act(self.bn1(self.conv1(x)))
            else:
                x = self.act(self.conv1(x))

        return x


class FullyConnected(nn.Module):
    """
    Creates an instance of a fully-connected layer. This includes the
    hidden layers but also the type of normalisation "batch" or
    "weight", the activation function, and initialises the weights.
    """
    def __init__(self, in_channels, out_channels, activation, normalisation,
                 att=None):
        super(FullyConnected, self).__init__()
        self.att = att
        self.norm = normalisation
        self.fc = nn.Linear(in_features=in_channels,
                            out_features=out_channels)
        if activation == 'sigmoid':
            self.act = nn.Sigmoid()
            self.norm = None
        elif activation == 'softmax':
            self.act = nn.Softmax(dim=-1)
            self.norm = None
        elif activation == 'global':
            self.act = None
            self.norm = None
        else:
            self.act = nn.ReLU()
            if self.norm == 'bn':
                self.bnf = nn.BatchNorm1d(out_channels)
            elif self.norm == 'wn':
                self.wnf = nn.utils.weight_norm(self.fc, name='weight')

        self.init_weights()

    def init_weights(self):
        """
        Initialises the weights of the current layer
        """
        if self.att:
            init_att_layer(self.fc)
        else:
            init_layer(self.fc)
        if self.norm == 'bn':
            init_bn(self.bnf)

    def forward(self, input):
        """
        Passes the input through the fully-connected layer

        Input
            input: torch.Tensor - The current input at this stage of the network
        """
        x = input
        if self.norm is not None:
            if self.norm == 'bn':
                x = self.act(self.bnf(self.fc(x)))
            else:
                x = self.act(self.wnf(x))
        else:
            if self.att:
                if self.act:
                    x = self.act(self.fc(x))
                else:
                    x = self.fc(x)
            else:
                if self.act:
                    x = self.act(self.fc(x))
                else:
                    x = self.fc(x)        

        return x


class ConvLSTM_Visual(nn.Module):
    def __init__(self, input_dim, output_dim, conv_hidden, lstm_hidden, num_layers, activation, norm, dropout):
        super(ConvLSTM_Visual, self).__init__()
        self.conv = ConvBlock2d(in_channels=input_dim,
                                out_channels=conv_hidden,
                                kernel=(72, 3),
                                stride=(1, 1),
                                pad=(0, 1),
                                normalisation='bn')
        self.pool = nn.MaxPool1d(kernel_size=3,
                                 stride=3,
                                 padding=0)
        self.drop = nn.Dropout(dropout)
        self.lstm = nn.LSTM(input_size=conv_hidden,
                            hidden_size=lstm_hidden,
                            num_layers=num_layers,
                            batch_first=True,
                            dropout=dropout,
                            bidirectional=True)
        self.fc = FullyConnected(in_channels=lstm_hidden*2,
                                 out_channels=output_dim,
                                 activation=activation,
                                 normalisation=norm)

    def forward(self, net_input):
        x = net_input
        batch, C, F, T = x.shape
        x = self.conv(x)
        x = self.pool(x.squeeze())
        x = self.drop(x)
        x = x.permute(0, 2, 1).contiguous()
        x, _ = self.lstm(x)                                 # output shape: (batch, width//stride(pool), lstm_hidden*2) 5x600x128
        x = self.fc(x[:, -1, :].reshape(batch, -1))         # output shape: (batch, output_dim)

        return x


class ConvLSTM_Audio(nn.Module):
    def __init__(self, input_dim, output_dim, conv_hidden, lstm_hidden, num_layers, activation, norm, dropout):
        super(ConvLSTM_Audio, self).__init__()
        self.conv = ConvBlock1d(in_channels=input_dim,      # 80
                                out_channels=conv_hidden,   # 128
                                kernel=3,
                                stride=1,
                                pad=1,
                                normalisation='bn')         # ['bn', 'wn', else]
        self.pool = nn.MaxPool1d(kernel_size=3,
                                 stride=3,
                                 padding=0)
        self.drop = nn.Dropout(dropout)                     # 0.2
        self.lstm = nn.LSTM(input_size=conv_hidden,         # 128
                            hidden_size=lstm_hidden,        # 128
                            num_layers=num_layers,          # 2
                            batch_first=True,
                            dropout=dropout,
                            bidirectional=True)
        self.fc = FullyConnected(in_channels=lstm_hidden*2,   # 128
                                 out_channels=output_dim,   # 2
                                 activation=activation,     # ['sigmoid', 'softmax', 'global', else]
                                 normalisation=norm)        # ['bn', 'wn']: nn.BatchNorm1d, nn.utils.weight_norm

    def forward(self, net_input):
        x = net_input
        batch, freq, width = x.shape
        x = self.conv(x)
        x = self.pool(x)
        x = self.drop(x)
        x = x.permute(0, 2, 1).contiguous()
        x, _ = self.lstm(x)                                 # output shape: (batch, width//stride(pool), lstm_hidden*2) 5x600x128
        x = self.fc(x[:, -1, :].reshape(batch, -1))         # output shape: (batch, output_dim)

        return x


class ConvLSTM_Text(nn.Module):
    def __init__(self, input_dim, output_dim, conv_hidden, lstm_hidden, num_layers, activation, norm, dropout):
        super(ConvLSTM_Text, self).__init__()
        self.conv = ConvBlock1d(in_channels=input_dim,
                                out_channels=conv_hidden,
                                kernel=3,
                                stride=1,
                                pad=1,
                                normalisation='bn')         # ['bn', 'wn', else]
        self.pool = nn.MaxPool1d(kernel_size=3,
                                 stride=3,
                                 padding=0)
        self.drop = nn.Dropout(dropout)
        self.lstm = nn.LSTM(input_size=conv_hidden,
                            hidden_size=lstm_hidden,
                            num_layers=num_layers,
                            batch_first=True,
                            dropout=dropout,
                            bidirectional=True)
        self.fc = FullyConnected(in_channels=lstm_hidden*2,
                                 out_channels=output_dim,
                                 activation=activation,     # ['sigmoid', 'softmax', 'global', else]
                                 normalisation=norm)        # ['bn', 'wn']: nn.BatchNorm1d, nn.utils.weight_norm

    def forward(self, net_input):
        x = net_input
        batch, F, T = x.shape
        x = self.conv(x)
        x = self.pool(x)
        x = self.drop(x)
        x = x.permute(0, 2, 1).contiguous()
        x, _ = self.lstm(x)                                 # output shape: (batch, width//stride(pool), lstm_hidden*2) 5x600x128
        x = self.fc(x[:, -1, :].reshape(batch, -1))         # output shape: (batch, output_dim)

        return x