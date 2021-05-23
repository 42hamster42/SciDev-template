from nn.base import Module
import numpy as np
import torch


class Softmax(Module):
    def softmax(self, x):
        x -= torch.max(x, dim=-1).values.unsqueeze(-1)
        exps = torch.exp(x)
        sumexps = torch.sum(exps, dim=-1).unsqueeze(-1)
        return exps / sumexps

    def forward(self, *args):
        [x] = args
        # assert (torch.abs(torch.sum(self.softmax(x), dim=-1) - 1) < 2e-4).all(), self.softmax(x)
        return self.softmax(x)

    def backward(self, x, grad_output):  # compute grad
        self.grad = self.softmax(x) * (1 - self.softmax(x)) * grad_output
        return self.grad


class CrossEntropy(Module):
    def cross_entropy(self, p, q):
        return -torch.sum(p * torch.log(q))

    def one_hot_target(self, prediction, target):
        if len(prediction.size()) == 1:
            prediction = prediction.reshape((1, prediction.size()[0]))
        result = torch.zeros_like(prediction)
        result[torch.arange(prediction.size()[0], dtype=torch.long), target.long()] = 1
        return result

    def forward(self, *args):
        [x, y] = args
        y = self.one_hot_target(x, y)
        return self.cross_entropy(y, x)

    def backward(self, x, y):
        y = self.one_hot_target(x, y)
        self.grad = -y / x
        return self.grad


class KLDivergence(Module):
    def kl_divergence(self, p, q):
        EPS = 1e-8
        clamped = torch.clamp(p, EPS, 1 - EPS)
        return torch.sum(p * (torch.log(clamped) - torch.log(q)))

    def one_hot_target(self, prediction, target):
        if len(prediction.size()) == 1:
            prediction = prediction.reshape((1, prediction.size()[0]))
        result = torch.zeros_like(prediction)
        result[torch.arange(prediction.size()[0], dtype=torch.long), target.long()] = 1
        return result

    def forward(self, *args):
        [x, y] = args
        y = self.one_hot_target(x, y)
        return self.kl_divergence(y, x)

    def backward(self, x, y):
        y = self.one_hot_target(x, y)
        self.grad = -y / x
        return self.grad


class MSE(Module):
    def forward(self, *args):
        [x, y] = args
        return torch.mean((x - y) ** 2)

    def backward(self, x, y):
        # coeff = 1.
        # if len(x.size()) > 1:
        #     coeff /= x.size()[0]
        return 2 * (x - y) / torch.numel(x)
