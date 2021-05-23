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
    _cell_types_to_cells = {
        'Basic': RNNCell,
        'LSTM': LSTMCell,
        'GRU': GRUCell
    }

    def __init__(self, input_size, output_size, cell_type=None, loss=None, optimizer=None):
        super(RNNModel, self).__init__()
        self.optimizer = optimizer
        self.cell_type = cell_type
        self.cell = self._cell_types_to_cells[cell_type](input_size, output_size, optimizer=optimizer)
        self.loss = loss
        self.output_size = output_size

    def forward(self, x):
        if self.cell_type != 'LSTM':
            hidden_state = torch.zeros(self.output_size)
            result = []
            self._hiddens = []
            for i in range(x.shape[0]):
                self._hiddens.append(hidden_state)
                hidden_state = self.cell.forward(x[i], hidden_state)
                result.append(hidden_state)
        else:
            assert False
        return torch.stack(result)

    def backward(self, x, y):
        predicted = self.forward(x)
        loss_value = self.loss.forward(predicted, y)
        grad_output = self.loss.backward(x, y)
        lg = torch.zeros(self.output_size)
        l = x.size()[0]
        for j in range(l):
            i = l - j - 1
            x_i = x[i]
            h_i = self._hiddens[i]
            grad_i = grad_output[i]
            (i_g, h_g) = self.cell.backward((x_i, h_i), lg + grad_i)
            lg = h_g
        return loss_value

    def zero_grad(self):
        self.cell.zero_grad()

    def apply_grad(self):
        self.cell.apply_grad()

    def train(self, data, n_epochs):
        x, y = data
        x = torch.tensor(x, dtype=torch.float)
        y = torch.tensor(y, dtype=torch.float)
        assert x.shape[0] == y.shape[0]
        dataset_length = y.shape[0]
        losses_history = []
        for epoch_n in range(n_epochs):
            loss_accum = 0
            for i in range(dataset_length):
                loss_accum += self.step(x[i], y[i])
            loss_accum /= dataset_length
            losses_history.append(loss_accum)

            print(f'Epoch {epoch_n}/{n_epochs}: loss={loss_accum}')
        plt.plot(np.arange(n_epochs) + 1, losses_history)
        plt.show()
