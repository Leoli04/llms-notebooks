import math
import torch
from torch import nn
from torch.optim import lr_scheduler
from d2l import torch as d2l

class SquareRootScheduler:
    def __init__(self, lr=0.1):
        self.lr = lr

    def __call__(self, num_update):
        return self.lr * pow(num_update + 1.0, -0.5)

class FactorScheduler:
    '''
     单因子调度器
    '''
    def __init__(self, factor=1, stop_factor_lr=1e-7, base_lr=0.1):
        self.factor = factor
        self.stop_factor_lr = stop_factor_lr
        self.base_lr = base_lr

    def __call__(self, num_update):
        self.base_lr = max(self.stop_factor_lr, self.base_lr * self.factor)
        return self.base_lr


if __name__ == '__main__':
    lr, num_epochs = 0.3, 30
    scheduler = SquareRootScheduler(lr=0.1)
    d2l.plot(torch.arange(num_epochs), [scheduler(t) for t in range(num_epochs)])