
import torch
from torch import nn
from d2l import torch as d2l

def relu(X):
    a = torch.zeros_like(X)
    return torch.max(X, a)

def net(X):
    '''
    自定义模型，一个隐藏层，一个输出层
    :param X:
    :return:
    '''
    X = X.reshape((-1, num_inputs))
    # 这里“@”代表矩阵乘法
    H = relu(X@W1 + b1)
    return (H@W2 + b2)

def train_custom(train_iter, test_iter,params):
    loss = nn.CrossEntropyLoss(reduction='none')
    num_epochs, lr = 10, 0.1
    updater = torch.optim.SGD(params, lr=lr)
    # 训练
    d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, updater)
    # 预测
    d2l.predict_ch3(net, test_iter)

def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.01)

def train_pytorch(train_iter, test_iter,params):
    # 定义模型
    net = nn.Sequential(nn.Flatten(),
                        nn.Linear(784, 256),
                        nn.ReLU(),
                        nn.Linear(256, 10))
    # 设置权重
    net.apply(init_weights);

    batch_size, lr, num_epochs = 256, 0.1, 10
    loss = nn.CrossEntropyLoss(reduction='none')
    trainer = torch.optim.SGD(net.parameters(), lr=lr)
    # 加载数据
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
    # 训练
    d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)

    # 预测
    d2l.predict_ch3(net, test_iter)
if __name__ == '__main__':
    batch_size = 256
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
    num_inputs, num_outputs, num_hiddens = 784, 10, 256

    W1 = nn.Parameter(torch.randn(
        num_inputs, num_hiddens, requires_grad=True) * 0.01)
    b1 = nn.Parameter(torch.zeros(num_hiddens, requires_grad=True))
    W2 = nn.Parameter(torch.randn(
        num_hiddens, num_outputs, requires_grad=True) * 0.01)
    b2 = nn.Parameter(torch.zeros(num_outputs, requires_grad=True))

    params = [W1, b1, W2, b2]

    train_custom(train_iter,test_iter,params)

    # train_pytorch(train_iter, test_iter, params)