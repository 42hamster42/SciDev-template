import torch
import numpy as np
from nn.base import Module


class FullyConnectedLayer(Module):
    def __init__(self, in_features, out_features, bias=True, init=None, optimizer=None):
        super(FullyConnectedLayer, self).__init__()
        self.optimizer = optimizer
        self._in_features = in_features
        self._out_features = out_features
        self._biases = None
        if init is not None:
            self._weights = init
            assert init.size()[0] == out_features
            assert init.size()[1] == in_features
        else:
            self._weights = torch.randn([out_features, in_features])
            self._weights *= np.sqrt(
                2 / (in_features + out_features))  # чтобы дисперсия на каждом слое была одинаковой
        if bias:
            self._biases = torch.randn(out_features)
            self._biases *= np.sqrt(2 / (out_features))

        self.grad = torch.zeros(in_features)
        self._grad_weights = torch.zeros((in_features, out_features))
        if bias:
            self._grad_bias = torch.zeros(out_features)

    def _has_biases(self):
        return self._biases is not None

    def forward(self, x):
        # if len(x.size()) == 1:
        #     x = torch.reshape(x, [1, x.shape[0]])
        y = x @ self._weights.T
        if self._has_biases():
            y += self._biases
        return y

    def backward(self, x, grad_output):
        extra_dim = False
        if len(grad_output.size()) == 1:
            assert len(x.size()) == 1
            grad_output = torch.reshape(grad_output, [1, grad_output.size()[0]])
            x = torch.reshape(x, [1, x.size()[0]])
            extra_dim = True
        self._grad_weights = grad_output.T @ x
        assert self._grad_weights.size() == self._weights.size()
        if self._has_biases():
            self._grad_bias = torch.sum(grad_output, dim=0)
            assert self._grad_bias.size() == self._biases.size()
        self.grad = grad_output @ self._weights
        if extra_dim:
            assert self.grad.size()[0]==1
            self.grad = self.grad[0]
        return self.grad

    def zero_grad(self):
        torch.zeros(self._grad_weights.size(), out=self._grad_weights)
        torch.zeros(self.grad.size(), out=self.grad)
        if self._has_biases():
            torch.zeros(self._grad_bias.size(), out=self._grad_bias)

    def apply_grad(self):
        if self.optimizer is None:
            raise ValueError('Need optimizer')
        self.optimizer.step(self._weights, self._grad_weights)
        if self._has_biases():
            self.optimizer.step(self._biases, self._grad_bias)
