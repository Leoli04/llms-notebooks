import torch
from torch import nn
from d2l import torch as d2l


from train_ch3  import train_ch3


def dropout_layer(X, dropout):
    '''

    :param X: 输入张量
    :param dropout: 暂退概率，为0表示不丢弃，为1 表示丢弃所有
    :return:
    '''
    assert 0 <= dropout <= 1
    # 在本情况中，所有元素都被丢弃
    if dropout == 1:
        return torch.zeros_like(X)
    # 在本情况中，所有元素都被保留
    if dropout == 0:
        return X
    # torch.rand(X.shape)：生成一个与X形状相同的张量，其中的元素是从0到1的均匀分布的随机数。
    # .float()方法将这个布尔张量转换为浮点数张量，其中True变为1.0，False变为0.0
    mask = (torch.rand(X.shape) > dropout).float()
    print(mask)
    return mask * X / (1.0 - dropout)

class Net(nn.Module):
    def __init__(self, num_inputs, num_outputs, num_hiddens1, num_hiddens2,
                 is_training = True):
        super(Net, self).__init__()
        self.num_inputs = num_inputs
        self.training = is_training
        self.lin1 = nn.Linear(num_inputs, num_hiddens1)
        self.lin2 = nn.Linear(num_hiddens1, num_hiddens2)
        self.lin3 = nn.Linear(num_hiddens2, num_outputs)
        self.relu = nn.ReLU()

    def forward(self, X):
        H1 = self.relu(self.lin1(X.reshape((-1, self.num_inputs))))
        # 只有在训练模型时才使用dropout
        if self.training == True:
            # 在第一个全连接层之后添加一个dropout层
            H1 = dropout_layer(H1, dropout1)
        H2 = self.relu(self.lin2(H1))
        if self.training == True:
            # 在第二个全连接层之后添加一个dropout层
            H2 = dropout_layer(H2, dropout2)
        out = self.lin3(H2)
        return out

def train_custom(train_iter, test_iter,num_inputs, num_outputs, num_hiddens1, num_hiddens2,loss,num_epochs):
    # 实例化模型
    net = Net(num_inputs, num_outputs, num_hiddens1, num_hiddens2)
    # 使用随机梯度下降优化器 优化模型参数
    trainer = torch.optim.SGD(net.parameters(), lr=lr)
    train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)

def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.01)

def train_pytorch(train_iter,test_iter,loss,num_epochs):
    net = nn.Sequential(nn.Flatten(),
                        nn.Linear(784, 256),
                        nn.ReLU(),
                        # 在第一个全连接层之后添加一个dropout层
                        nn.Dropout(dropout1),
                        nn.Linear(256, 256),
                        nn.ReLU(),
                        # 在第二个全连接层之后添加一个dropout层
                        nn.Dropout(dropout2),
                        nn.Linear(256, 10))
    net.apply(init_weights);
    trainer = torch.optim.SGD(net.parameters(), lr=lr)
    train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)

if __name__ == '__main__':

    # sys.path.append(str(Path(__file__).resolve().parents[1]))
    ####################################
    # 模型输入
    ####################################
    X = torch.arange(16, dtype=torch.float32).reshape((2, 8))
    print(X)
    # print(dropout_layer(X, 0.))
    # print(dropout_layer(X, 0.5))
    # print(dropout_layer(X, 1.))

    ####################################
    # 模型输入数据
    ####################################
    num_epochs, lr, batch_size = 10, 0.5, 256
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

    ####################################
    # 模型参数
    ####################################
    num_inputs, num_outputs, num_hiddens1, num_hiddens2 = 784, 10, 256, 256
    dropout1, dropout2 = 0.2, 0.5

    loss = nn.CrossEntropyLoss(reduction='none')

    # train_custom(train_iter,test_iter,num_inputs,num_outputs,num_hiddens1,num_hiddens2,loss,num_epochs)

    train_pytorch(train_iter,test_iter,loss,num_epochs)

