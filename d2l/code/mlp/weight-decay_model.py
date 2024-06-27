
import torch
from torch import nn
from d2l import torch as d2l

def init_params():
    '''
    随机初始化模型参数
    :return:
    '''
    w = torch.normal(0, 1, size=(num_inputs, 1), requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    return [w, b]

def l2_penalty(w):
    '''
    定义L2范数惩罚
    :param w:
    :return:
    '''
    # 对所有权重项求平方后并将它们求和
    return torch.sum(w.pow(2)) / 2

def train_custom(lambd):
    '''

    :param lambd: 权重衰减系数，为0时，不适用权重衰减
    :return:
    '''
    # 初始化模型参数
    w, b = init_params()
    net, loss = lambda X: d2l.linreg(X, w, b), d2l.squared_loss
    num_epochs, lr = 100, 0.003
    animator = d2l.Animator(xlabel='epochs', ylabel='loss', yscale='log',
                            xlim=[5, num_epochs], legend=['train', 'test'])
    for epoch in range(num_epochs):
        for X, y in train_iter:
            # 增加了L2范数惩罚项，
            # 广播机制使l2_penalty(w)成为一个长度为batch_size的向量
            l = loss(net(X), y) + lambd * l2_penalty(w)
            l.sum().backward()
            d2l.sgd([w, b], lr, batch_size)
        if (epoch + 1) % 5 == 0:
            animator.add(epoch + 1, (d2l.evaluate_loss(net, train_iter, loss),
                                     d2l.evaluate_loss(net, test_iter, loss)))
    print('w的L2范数是：', torch.norm(w).item())

def train_concise(wd):
    '''
    通过框架实现权重衰减
    :param wd: 权重衰减系数
    :return:
    '''
    net = nn.Sequential(nn.Linear(num_inputs, 1))
    for param in net.parameters():
        # 将param中的数据用标准正态分布（均值为0，标准差为1）随机值填充。
        param.data.normal_()
    # 使用均方误差计算损失函数
    loss = nn.MSELoss(reduction='none')
    num_epochs, lr = 100, 0.003
    # 偏置参数没有衰减
    # 使用随机梯度下降算法的优化器
    trainer = torch.optim.SGD([
        # net[0].weight（即网络第一层的权重），并为这个参数组设置了权重衰减 wd
        {"params":net[0].weight,'weight_decay': wd},
        # net[0].bias（即网络第一层的偏置
        {"params":net[0].bias}], lr=lr)
    animator = d2l.Animator(xlabel='epochs', ylabel='loss', yscale='log',
                            xlim=[5, num_epochs], legend=['train', 'test'])
    for epoch in range(num_epochs):
        for X, y in train_iter:
            trainer.zero_grad()
            l = loss(net(X), y)
            l.mean().backward()
            trainer.step()
        if (epoch + 1) % 5 == 0:
            animator.add(epoch + 1,
                         (d2l.evaluate_loss(net, train_iter, loss),
                          d2l.evaluate_loss(net, test_iter, loss)))
    print('w的L2范数：', net[0].weight.norm().item())

'''
演示权重衰减
权重衰减通过在模型的损失函数中添加一个正则化项来实现，该正则化项与模型的权重参数的平方和成正比。
'''
if __name__ == '__main__':
    n_train, n_test, num_inputs, batch_size = 20, 100, 200, 5
    # 原始权重、偏置参数
    true_w, true_b = torch.ones((num_inputs, 1)) * 0.01, 0.05
    # 生成训练数据
    train_data = d2l.synthetic_data(true_w, true_b, n_train)
    train_iter = d2l.load_array(train_data, batch_size)
    # 生成测试数据
    test_data = d2l.synthetic_data(true_w, true_b, n_test)
    test_iter = d2l.load_array(test_data, batch_size, is_train=False)

    # train_custom(lambd=3)
    # 框架实现
    train_concise(3)