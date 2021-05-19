from nn.base import Module
import numpy as np
import torch


class Softmax(Module):
    def forward(self, *args):
        [x] = args
        x -= torch.max(x)
        exps = torch.exp(x)
        sumexps = torch.sum(exps)
        return exps / sumexps

    def backward(self, x, grad_output):  # compute grad
        raise NotImplementedError()


class CrossEntropy(Module):
    pass  # TODO: replace line with your code


class KLDivergence(Module):
    pass  # TODO: replace line with your code


class MSE(Module):
    def forward(self, *args):
        [x, y] = args
        return torch.mean((x - y) ** 2)

    def backward(self, x, y):
        coeff = 1.
        if len(x.size()) > 1:
            coeff /= x.size()[0]
        return 2 * (x - y) * coeff
