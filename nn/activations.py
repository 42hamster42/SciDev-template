from nn.base import Module
import torch


class Sigmoid(Module):
    def __init__(self):
        self.grad = None

    def _sigmoid(self, x):
        return 1 / (1 + torch.exp(-x))

    def forward(self, *args):
        assert len(args) == 1
        return self._sigmoid(args[0])

    def backward(self, x, grad_output):  # compute grad
        assert self.grad is None
        self.grad = self._sigmoid(x) * (1 - self._sigmoid(x)) * grad_output
        return self.grad

    def zero_grad(self):
        self.grad = None

    def apply_grad(self):
        # Ничего не делаем
        pass


class ReLU(Module):
    def __init__(self):
        self.grad = None

    def _relu(self, x):
        return x * (x > 0)

    def forward(self, *args):
        assert len(args) == 1
        return self._relu(args[0])

    def backward(self, x, grad_output):  # compute grad
        assert self.grad is None
        self.grad = (x > 0) * grad_output
        return self.grad

    def zero_grad(self):
        self.grad = None

    def apply_grad(self):
        # Ничего не делаем
        pass
