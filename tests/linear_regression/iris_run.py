import torch

from nn.layers import FullyConnectedLayer
from nn.activations import ReLU, Sigmoid
from nn.models import FeedForwardModel
from nn.losses import MSE, Softmax, CrossEntropy, KLDivergence
from nn.optimizer import GradientDescend, GradientDescendWithMomentum, GradientDescendWithNesterovMomentum
from nn.data import get_iris
from matplotlib import pyplot as plt


def main():
    torch.manual_seed(1)
    data = get_iris()

    [x, y] = data
    print(x.size())
    print(y.size())

    # model = FeedForwardModel(
    #     layers=[
    #         FullyConnectedLayer(
    #             4, 15, bias=True,
    #         ),
    #         # ReLU(),
    #         Sigmoid(),
    #         FullyConnectedLayer(
    #             15, 5, bias=True,
    #         ),
    #         # ReLU(),
    #         Sigmoid(),
    #         FullyConnectedLayer(
    #             5, 3, bias=True
    #         ),
    #         Softmax()
    #     ],
    #     loss=CrossEntropy(),
    #     optimizer = GradientDescend(lr=0.0001)
    # #     optimizer=GradientDescendWithNesterovMomentum(lr=0.0001, moment=0.8)
    # )

    model = FeedForwardModel(
        layers=[
            FullyConnectedLayer(
                4, 3, bias=True,
            ),
            Softmax()
        ],
        loss=CrossEntropy(),
        optimizer=GradientDescendWithNesterovMomentum(lr=0.001, moment=0.9)
    # optimizer = GradientDescend(lr=0.001)
    )

    def train_acc(model, x, y):
        predicted = model.forward(x)
        predicted = predicted.max(dim=-1).indices
        predicted_r = predicted.reshape(-1, 1)
        return torch.sum(predicted_r == y).item() / torch.numel(y)

    model.train(data, n_epochs=300, batch_size=10, metric=train_acc)

    x, y = data
    print(train_acc(model, x, y))


if __name__ == '__main__':
    with torch.no_grad():
        main()
