import torch

from nn.activations import Sigmoid, ReLU
from nn.layers import FullyConnectedLayer
from nn.models import FeedForwardModel
from nn.losses import MSE
from nn.optimizer import GradientDescend
from nn.data import get_line
from matplotlib import pyplot as plt

def linear_regression():
    torch.manual_seed(1)
    data = get_line(seed=1)
    y = data[1]
    # y += 1
    model = FeedForwardModel(
        layers=[
            FullyConnectedLayer(
                1, 1, bias=False,
                init=torch.full((1, 1), 0.5, dtype=torch.float)
            ),
        ],
        loss=MSE(),
        optimizer=GradientDescend(lr=0.3)
    )
    model.train(data, n_epochs=20, batch_size=100)
    plt.scatter(*data, color='b')
    x, y = data
    predicted = model.forward(x)
    plt.plot(x, predicted, color='r')
    plt.show()



def regression_net():
    torch.manual_seed(1)
    data = get_line(seed=1)
    y = data[1]
    y += 1
    model = FeedForwardModel(
        layers=[
            FullyConnectedLayer(
                1, 1, bias=True,
                init=torch.full((1, 1), 0.5, dtype=torch.float)
            ),

            FullyConnectedLayer(
                1, 5, bias=True,
            ),
            Sigmoid(),
            FullyConnectedLayer(
                5, 3, bias=True,
            ),
            Sigmoid(),
            FullyConnectedLayer(
                3, 1, bias=True,
            ),

        ],
        loss=MSE(),
        optimizer=GradientDescend(lr=0.3)
    )
    model.train(data, n_epochs=500, batch_size=10)
    plt.scatter(*data, color='b')
    x, y = data
    x_sorted, _ = torch.sort(x, dim=0)
    predicted = model.forward(x_sorted)
    plt.scatter(x_sorted, predicted, color='r')
    plt.plot(x_sorted, predicted, color='r')
    plt.show()

def main():
    linear_regression()


if __name__ == '__main__':
    with torch.no_grad():
        main()
