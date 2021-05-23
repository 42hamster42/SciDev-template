import torch

from nn.activations import Tanh
from nn.base import Module

# You can use your implementation of FullyConnectedLayer (Linear)
from nn.layers import FullyConnectedLayer

'''
These cells accept single element of sequence and calculate output hidden states.
Target output is calculated by one of the RNNModel.
'''


class RNNCell(Module):
    def __init__(self, input_size, hidden_size, init=None, optimizer=None):
        super(RNNCell, self).__init__()
        init_i = None
        init_h = None
        if init:
            (init_i, init_h) = init
        self.input_gate = FullyConnectedLayer(input_size, hidden_size, init=init_i, bias=True, optimizer=optimizer)
        self.hidden_gate = FullyConnectedLayer(hidden_size, hidden_size, init=init_h, bias=True, optimizer=optimizer)
        self.activation = Tanh()

    def forward(self, x, h):
        ig_result = self.input_gate.forward(x)
        hg_result = self.input_gate.forward(h)
        sum = ig_result + hg_result
        return self.activation.forward(sum)

    def backward(self, x, grad_output):
        (x, h) = x
        ig_result = self.input_gate.forward(x)
        hg_result = self.input_gate.forward(h)
        sum = ig_result + hg_result
        tanh_grad = self.activation.backward(sum, grad_output)
        self.input_grad = self.input_gate.backward(x, tanh_grad)
        self.hidden_grad = self.hidden_gate.backward(h, tanh_grad)
        return (self.input_grad, self.hidden_grad)

    def zero_grad(self):
        self.input_gate.zero_grad()
        self.hidden_gate.zero_grad()
        self.activation.zero_grad()

    def apply_grad(self):
        self.input_gate.apply_grad()
        self.hidden_gate.apply_grad()
        self.activation.apply_grad()


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
