import random
import torch
from torch.utils import data
from torch import nn

def synthetic_data(w, b, num_examples):
    """生成y=Xw+b+噪声"""

    # 返回一个形状为 (num_examples, len(w)) 的张量，其中的每个元素都是从均值为0、标准差为1的正态分布中随机生成的。
    X = torch.normal(0, 1, (num_examples, len(w)))
    # x是二维，w是一位，相当于做矩阵向量机，返回一位张量，长度为1000，可以理解为1000个样本，生成1000结果
    y = torch.matmul(X, w) + b
    y += torch.normal(0, 0.01, y.shape)
    # x.shape为1000*2，把y转成1000*1的张量
    return X, y.reshape((-1, 1))

def data_iter(batch_size, features, labels):
    '''
    数据迭代器：
    先根据样本数量给每个样本标号
    然后打乱顺序
    在根据批次，把样本、标签分成多个批次
    :param batch_size: 批量大小
    :param features: 特征
    :param labels: 标签
    :return:
    '''
    # 样本数量
    num_examples = len(features)
    # 给样本标号
    indices = list(range(num_examples))
    # 这些样本是随机读取的，没有特定的顺序
    random.shuffle(indices)
    for i in range(0, num_examples, batch_size):
        batch_indices = torch.tensor(
            indices[i: min(i + batch_size, num_examples)])
        yield features[batch_indices], labels[batch_indices]

def linreg(X, w, b):  #@save
    """线性回归模型"""
    return torch.matmul(X, w) + b

def squared_loss(y_hat, y):  #@save
    '''
    均方损失
    :param y_hat: 批量预测值
    :param y: 批量实际值
    :return:
    '''
    return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2

def sgd(params, lr, batch_size):  #@save
    '''
    小批量随机梯度下降
    :param params: 权重和偏置组成的向量/数组
    :param lr: 学习率
    :param batch_size:
    :return:
    '''
    # 关闭梯度计算
    with torch.no_grad():
        for param in params:
            param -= lr * param.grad / batch_size
            param.grad.zero_()

def train(net,w,b,lr,num_epochs ,loss,features, labels ,batch_size):
    '''
    训练模型
    :param net: 模型函数
    :param w: 权重
    :param b: 偏置
    :param lr: 学习率，
    :param num_epochs: 训练批次
    :param loss: 损失函数
    :param features: 特征
    :param labels: 标签
    :batch_size : 批量大小
    :return:
    '''
    for epoch in range(num_epochs):
        for X, y in data_iter(batch_size, features, labels):
            l = loss(net(X, w, b), y)  # X和y的小批量损失
            # 因为l形状是(batch_size,1)，而不是一个标量。l中的所有元素被加到一起，
            # 并以此计算关于[w,b]的梯度
            l.sum().backward()
            sgd([w, b], lr, batch_size)  # 使用参数的梯度更新参数
        # 每轮轮训练完成，关闭梯度计算，查看平均损失
        with torch.no_grad():
            train_l = loss(net(features, w, b), labels)
            print(f'epoch {epoch + 1}, loss {float(train_l.mean()):f}')

def load_array(data_arrays, batch_size, is_train=True):  #@save
    """
    构造一个PyTorch数据迭代器
    :param data_arrays:
    :param batch_size:
    :param is_train:
    :return:
    """
    dataset = data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset, batch_size, shuffle=is_train)



def train_custom(features, labels,batch_size):
    '''
    通过自定义方式实现
    :param features:
    :param labels:
    :param batch_size:
    :return:
    '''
    w = torch.normal(0, 0.01, size=(2, 1), requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    lr = 0.03
    num_epochs = 3
    net = linreg
    loss = squared_loss

    train(net,w,b,lr,num_epochs,loss,features, labels,batch_size)

def train_pytorch(features, labels,batch_size):
    '''
    通过pytorch框架实现
    :param features:
    :param labels:
    :param batch_size:
    :return:
    '''
    data_iter = load_array((features, labels), batch_size)
    # 指定输入特征形状，输出特征形状
    net = nn.Sequential(nn.Linear(2, 1))

    # 权重及偏置初始化
    # net[0]选择网络中的第一个图层
    net[0].weight.data.normal_(0, 0.01)
    net[0].bias.data.fill_(0)

    #  损失函数
    loss = nn.MSELoss()
    # 优化算法
    trainer = torch.optim.SGD(net.parameters(), lr=0.03)

    num_epochs = 3
    for epoch in range(num_epochs):
        for X, y in data_iter:
            l = loss(net(X), y)
            trainer.zero_grad()
            l.backward()
            trainer.step()
        l = loss(net(features), labels)
        print(f'epoch {epoch + 1}, loss {l:f}')

if __name__ == '__main__':
    # 获取数据
    true_w = torch.tensor([2, -3.4])
    print(len(true_w))
    true_b = 4.2
    features, labels = synthetic_data(true_w, true_b, 1000)
    print('features:', features[0],'\nlabel:', labels[0])


    # 参数设置
    batch_size = 10
    # train_custom(features,labels,batch_size)
    # train_pytorch(features,labels,batch_size)
