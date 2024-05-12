import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class ConvLSTMCell(nn.Module):
    def __init__(self, input_size, input_dim, hidden_dim, kernel_size, bias=True, activation=F.tanh,
                 batch_norm=True, dropout=0.2, peepholes=False):
        """
        :param input_size: size of input tensor
        Args:
            input_size:
            input_dim:
            hidden_dim:
            kernel_size:
            bias:
            activation:
            batch_norm:
            dropout:
            peepholes:
        """
        super(ConvLSTMCell, self).__init__()
        self.height, self.width = input_size
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias = bias
        self.activation = activation
        self.batch_norm = batch_norm
        self.dropout = dropout
        self.peepholes = peepholes

        if peepholes:
            self.W_ci = nn.Parameter(torch.FloatTensor(hidden_dim, self.height, self.width))


class ConvLSTM(nn.Module):