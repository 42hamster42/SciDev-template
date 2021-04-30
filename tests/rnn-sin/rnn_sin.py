import numpy as np
import matplotlib.pyplot as plt
import torch

from nn.rnn_models import RNNModel

cell_types = ['Basic', 'LSTM', 'GRU']
seq_lengths = [2, 4, 8]


def create_data():
    Xmax = 4 * np.pi
    Xmin = 0
    Npoints = 200
    x = np.linspace(Xmin, Xmax, Npoints)
    y_true = np.sin(x)
    return x, y_true


def create_dataset(x, y, seq_length):
    x_s, y_s = [], []
    for i in range(len(x)-seq_length):
        seq_x = x[i:i+seq_length]
        target_y = y[i+seq_length]
        x_s.append(seq_x)
        y_s.append(target_y)
    return np.array(x_s)[..., None], np.array(y_s)[..., None]


def main():
    x_base, y_base = create_data()
    for cell_type in cell_types:
        for seq_len in seq_lengths:
            x_data, y_data = create_dataset(x_base, y_base, seq_len)
            model = RNNModel(1, cell_type=cell_type)
            #model.train()
            #model.forward()


if __name__ == '__main__':
    main()