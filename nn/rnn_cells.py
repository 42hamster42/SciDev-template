import torch
from nn.base import Module

#You can use your implementation of FullyConnectedLayer (Linear)
from nn.layers import FullyConnectedLayer


# Don't know how template will behave if I put this activation to activations.py
# If it is possible or if you want it can be moved
class Tanh(Module):
    pass  # TODO: replace line with your implementation


'''
These cells accept single element of sequence and calculate output hidden states.
Target output is calculated by one of the RNNModel.
'''


class RNNCell(Module):
    def __init__(self, input_size, hidden_size, init=None, optimizer=None):
        super(RNNCell, self).__init__()
        raise NotImplementedError

    def forward(self, x, h):
        raise NotImplementedError

#   x = cat([x, h])
    def backward(self, x, grad_output):
        raise NotImplementedError

    def zero_grad(self):
        raise NotImplementedError

    def apply_grad(self):
        raise NotImplementedError


class LSTMCell(Module):
    def __init__(self, input_size, hidden_size, init=None, optimizer=None):
        super(LSTMCell, self).__init__()
        raise NotImplementedError

    def forward(self, x, h_c):
        raise NotImplementedError

#   x = cat([x, h, c])
    def backward(self, x, grad_output):
        raise NotImplementedError

    def zero_grad(self):
        raise NotImplementedError

    def apply_grad(self):
        raise NotImplementedError


class GRUCell(Module):
    def __init__(self, input_size, hidden_size, init=None, optimizer=None):
        super(GRUCell, self).__init__()
        raise NotImplementedError

    def forward(self, x, h):
        raise NotImplementedError

#   x = cat([x, h])
    def backward(self, x, grad_output):
        raise NotImplementedError

    def zero_grad(self):
        raise NotImplementedError

    def apply_grad(self):
        raise NotImplementedError