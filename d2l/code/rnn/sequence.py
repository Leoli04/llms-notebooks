import torch
from torch import nn
from d2l import torch as d2l

# 初始化网络权重的函数
def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)

# 一个简单的多层感知机
def get_net():
    net = nn.Sequential(nn.Linear(4, 10),
                        nn.ReLU(),
                        nn.Linear(10, 1))
    net.apply(init_weights)
    return net

def train(net, train_iter, loss, epochs, lr):
    trainer = torch.optim.Adam(net.parameters(), lr)
    for epoch in range(epochs):
        for X, y in train_iter:
            trainer.zero_grad()
            l = loss(net(X), y)
            l.sum().backward()
            trainer.step()
        print(f'epoch {epoch + 1}, '
              f'loss: {d2l.evaluate_loss(net, train_iter, loss):f}')




if __name__ == '__main__':
    ################################
    # 生成数据
    ################################
    T = 1000  # 总共产生1000个点
    time = torch.arange(1, T + 1, dtype=torch.float32)
    x = torch.sin(0.01 * time) + torch.normal(0, 0.2, (T,))
    d2l.plot(time, [x], 'time', 'x', xlim=[1, 1000], figsize=(6, 3))

    ################################
    # 处理数据
    ################################
    # 前4个输入，对当前预测产生影响
    tau = 4
    features = torch.zeros((T - tau, tau))
    for i in range(tau):
        # x是一维张量，features是二维张量(t-tau,tau),将features的每列赋值为0/1/2/3-(T-tau+0/1/2/3)
        features[:, i] = x[i: T - tau + i]
    labels = x[tau:].reshape((-1, 1))

    batch_size, n_train = 16, 600
    # 只有前n_train个样本用于训练
    train_iter = d2l.load_array((features[:n_train], labels[:n_train]),
                                batch_size, is_train=True)
    ################################
    # 训练
    ################################

    # 平方损失。注意：MSELoss计算平方误差时不带系数1/2
    loss = nn.MSELoss(reduction='none')
    net = get_net()
    train(net, train_iter, loss, 5, 0.01)

    multistep_preds = torch.zeros(T)
    multistep_preds[: n_train + tau] = x[: n_train + tau]
    # 600+4 后面的数据都使用预测数据
    for i in range(n_train + tau, T):
        multistep_preds[i] = net(
            multistep_preds[i - tau:i].reshape((1, -1)))