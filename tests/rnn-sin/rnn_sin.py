import numpy as np
import matplotlib.pyplot as plt
import torch

from nn.losses import MSE
from nn.optimizer import GradientDescend
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
    for i in range(len(x) - seq_length):
        seq_x = x[i:i + seq_length]
        target_y = y[i + seq_length]
        x_s.append(seq_x)
        y_s.append(target_y)
    return np.array(x_s)[..., None], np.array(y_s)[..., None]


def main():
    x_base, y_base = create_data()

    x_base_t = torch.tensor(x_base, dtype=torch.float).unsqueeze(1)
    y_base_t = torch.tensor(y_base, dtype=torch.float).unsqueeze(1)

    for cell_type in cell_types:
        for seq_len in seq_lengths:
            x_data, y_data = create_dataset(x_base, y_base, seq_len)
            print(np.array(x_data).shape)
            model = RNNModel(1, 1, cell_type=cell_type, loss=MSE(), optimizer=GradientDescend(lr=0.001))
            model.train([x_data, y_data], 20)
            pred = model.forward(x_base_t)
            print(MSE().forward(pred, y_base_t))


if __name__ == '__main__':
    main()
