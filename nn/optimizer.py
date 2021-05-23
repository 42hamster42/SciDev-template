import torch

from nn.base import Optimizer


class GradientDescend(Optimizer):
    def __init__(self, lr):
        super(GradientDescend, self).__init__()
        self._lr = lr

    def step(self, weights, grad):
        weights -= grad * self._lr


class GradientDescendWithMomentum(GradientDescend):
    def __init__(self, lr, moment):
        super(GradientDescend, self).__init__()
        self._lr = lr
        self.moment = moment
        self.mov_average_gradients = dict()

    def step(self, weights, grad):
        mov_average_gradient = self.mov_average_gradients.get(id(weights), torch.zeros_like(grad))
        # mov_average_gradient = self.mov_average_gradient * self.moment + (1 - self.moment) * self._lr * grad
        mov_average_gradient *= self.moment
        mov_average_gradient += (1 - self.moment) * self._lr * grad
        weights -= mov_average_gradient
        self.mov_average_gradients[id(weights)] = mov_average_gradient


class GradientDescendWithNesterovMomentum(GradientDescend):
    def __init__(self, lr, moment):
        super(GradientDescend, self).__init__()
        self._lr = lr
        self.moment = moment
        self.mov_average_gradients = dict()
        self.moment_mode = True

    def model_step(self, model, x, y):
        self.moment_mode = True
        model.apply_grad()
        self.moment_mode = False
        model.zero_grad()
        loss = model.backward(x, y)
        model.apply_grad()
        return loss

    def step(self, weights, grad):
        mov_average_gradient = self.mov_average_gradients.get(id(weights), torch.zeros_like(weights))
        if self.moment_mode:
            mov_average_gradient *= self.moment
            weights -= mov_average_gradient
        else:
            current_shift = (1 - self.moment) * self._lr * grad
            mov_average_gradient += current_shift
            weights -= current_shift
        self.mov_average_gradients[id(weights)] = mov_average_gradient
