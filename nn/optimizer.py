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


# class GradientDescendWithNesterovMomentum(Optimizer):
#     def __init__(self, lr):
#         super(GradientDescend, self).__init__()
#         self._lr = lr
#
#         # raise NotImplementedError  # TODO: replace line with your code
#
#     def step(self, weights, grad):
#         self.mov_average_gradient = self.mov_average_gradient * self.moment + (1 - self.moment) * self._lr * GRAD  ###################
#         weights -= self.mov_average_gradient
