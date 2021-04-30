import numpy as np, pandas as pd, matplotlib.pyplot as plt

import torch
from nn.base import Model
from nn.rnn_cells import RNNCell, LSTMCell, GRUCell

'''
RNNModel which uses one of the RNNCells by cell_type.
"cell_type" should be one of the following: "Basic", "LSTM", "GRU".
Calculation of the y^{hat} output (outputs) based on hidden state should be done here. 
'''


class RNNModel(Model):
    def __init__(self, output_size, cell_type=None, loss=None, optimizer=None):
        super(RNNModel, self).__init__()
        raise NotImplementedError  # TODO: replace line with your implementation

    def forward(self, x):
        raise NotImplementedError  # TODO: replace line with your implementation

    def backward(self, x, y):
        raise NotImplementedError  # TODO: replace line with your implementation

    def zero_grad(self):
        raise NotImplementedError  # TODO: replace line with your implementation

    def apply_grad(self):
        raise NotImplementedError  # TODO: replace line with your implementation

    def train(self, data, n_epochs):
        raise NotImplementedError  # TODO: replace line with your implementation

