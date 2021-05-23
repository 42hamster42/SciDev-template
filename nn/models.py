import numpy as np, pandas as pd, matplotlib.pyplot as plt
import copy

from nn.base import Model


class FeedForwardModel(Model):
    def __init__(self, layers=None, loss=None, optimizer=None):
        super(FeedForwardModel, self).__init__()
        self.layers = copy.copy(layers)
        self.loss = loss
        self.optimizer = optimizer
        if optimizer:
            for layer in layers:
                if hasattr(layer,'optimizer') and not layer.optimizer:
                    layer.optimizer = optimizer
        self._forward_results = []

    def forward(self, x):
        self._forward_results = [x]
        for layer in self.layers:
            x = layer.forward(x)
            self._forward_results.append(x)
        return x

    def backward(self, x, y):
        predicted = self.forward(x)
        assert len(self._forward_results) == len(self.layers) + 1
        loss_value = self.loss.forward(predicted, y)
        grad_output = self.loss.backward(predicted, y)
        intemediate_values = self._forward_results[:-1]
        for v, layer in zip(intemediate_values[::-1], self.layers[::-1]):
            grad_output = layer.backward(v, grad_output)
        return loss_value

    def zero_grad(self):
        for layer in self.layers:
            layer.zero_grad()

    def apply_grad(self):
        for layer in self.layers:
            layer.apply_grad()

    def train(self, data, n_epochs, batch_size=None, metric=None):
        x, y = data
        assert x.shape[0] == y.shape[0]
        dataset_length = y.shape[0]
        losses_history = []
        for epoch_n in range(n_epochs):
            loss_accum = 0
            if batch_size:
                for start in range(0, dataset_length, batch_size):
                    end = np.min([start + batch_size, dataset_length])
                    loss_accum += self.step(x[start:end], y[start:end])
                loss_accum /= (dataset_length + batch_size - 1) // batch_size
            else:
                for i in range(dataset_length):
                    loss_accum += self.step(x[i], y[i])
                loss_accum /= dataset_length
            losses_history.append(loss_accum)

            metric_value = ''
            if metric:
                metric_value = f' metric={metric(self,x,y)}'

            print(f'Epoch {epoch_n}/{n_epochs}: loss={loss_accum}' + metric_value)
        plt.plot(np.arange(n_epochs) + 1, losses_history)
        plt.show()
