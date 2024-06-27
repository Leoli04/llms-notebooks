import torch
from torch import nn
from torch.nn import functional as F

'''
自定义模型，又不同方式的实现
'''

class MLP(nn.Module):
    '''
    模型定义，集成nn.Module
    '''
    # 用模型参数声明层。这里，我们声明两个全连接的层
    def __init__(self):
        # 调用MLP的父类Module的构造函数来执行必要的初始化。
        # 这样，在类实例化时也可以指定其他函数参数，例如模型参数params（稍后将介绍）
        super().__init__()
        self.hidden = nn.Linear(20, 256)  # 隐藏层
        self.out = nn.Linear(256, 10)  # 输出层

    # 定义模型的前向传播，即如何根据输入X返回所需的模型输出
    def forward(self, X):
        # 注意，这里我们使用ReLU的函数版本，其在nn.functional模块中定义。
        # 先经过隐藏层，在隐藏层结果经过relu激活函数得到输出
        return self.out(F.relu(self.hidden(X)))

class MySequential(nn.Module):
    '''
    顺序块：把其他模块串起来
    '''
    def __init__(self, *args):
        super().__init__()
        for idx, module in enumerate(args):
            # 这里，module是Module子类的一个实例。我们把它保存在'Module'类的成员
            # 变量_modules中。_module的类型是OrderedDict
            self._modules[str(idx)] = module

    def forward(self, X):
        # OrderedDict保证了按照成员添加的顺序遍历它们
        for block in self._modules.values():
            X = block(X)
        return X

class FixedHiddenMLP(nn.Module):
    def __init__(self):
        '''
        固定隐藏层
        '''
        super().__init__()
        # 不计算梯度的随机权重参数。因此其在训练期间保持不变
        self.rand_weight = torch.rand((20, 20), requires_grad=False)
        self.linear = nn.Linear(20, 20)

    def forward(self, X):
        '''
        :param X:
        :return:
        '''
        X = self.linear(X)
        # 使用创建的常量参数以及relu和mm函数
        X = F.relu(torch.mm(X, self.rand_weight) + 1)
        # 复用全连接层。这相当于两个全连接层共享参数
        X = self.linear(X)
        # 控制流
        while X.abs().sum() > 1:
            X /= 2
        return X.sum()

class NestMLP(nn.Module):
    '''
    嵌入模型
    '''
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(20, 64), nn.ReLU(),
                                 nn.Linear(64, 32), nn.ReLU())
        self.linear = nn.Linear(32, 16)

    def forward(self, X):
        return self.linear(self.net(X))


class CenteredLayer(nn.Module):
    '''
    自定义层
    '''
    def __init__(self):
        super().__init__()

    def forward(self, X):
        return X - X.mean()

class MyLinear(nn.Module):
    '''
    带参数的层
    '''
    def __init__(self, in_units, units):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(in_units, units))
        self.bias = nn.Parameter(torch.randn(units,))
    def forward(self, X):
        linear = torch.matmul(X, self.weight.data) + self.bias.data
        return F.relu(linear)

if __name__ == '__main__':
    net = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 1))
    X = torch.rand(size=(2, 4))

    # 模型参数访问
    print(net[2].state_dict())
    # 目标参数
    print(type(net[2].bias))
    print(net[2].bias)
    print(net[2].bias.data)

    # 一次性访问所有参数
    print(*[(name, param.shape) for name, param in net[0].named_parameters()])
    print(*[(name, param.shape) for name, param in net.named_parameters()])