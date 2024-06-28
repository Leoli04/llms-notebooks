
import torch
from torch import nn
from torch.nn import functional as F


def lenet():
    '''
    LeNet是最早发布的卷积神经网络之一

    LeNet（LeNet-5）由两个部分组成：
        卷积编码器：由两个卷积层组成;
        全连接层密集块：由三个全连接层组成。
    每个卷积块中的基本单元是一个卷积层、一个sigmoid激活函数和平均汇聚层
    '''
    net = nn.Sequential(
        nn.Conv2d(1, 6, kernel_size=5, padding=2), nn.Sigmoid(),
        nn.AvgPool2d(kernel_size=2, stride=2),
        nn.Conv2d(6, 16, kernel_size=5), nn.Sigmoid(),
        nn.AvgPool2d(kernel_size=2, stride=2),
        nn.Flatten(),
        nn.Linear(16 * 5 * 5, 120), nn.Sigmoid(),
        nn.Linear(120, 84), nn.Sigmoid(),
        nn.Linear(84, 10))
    return net

def  alexnet():
    '''
    alexnet给人的启发：从浅层网络到深层网络，
    :return:
    '''
    net = nn.Sequential(
        # 这里使用一个11*11的更大窗口来捕捉对象。
        # 同时，步幅为4，以减少输出的高度和宽度。
        # 另外，输出通道的数目远大于LeNet
        nn.Conv2d(1, 96, kernel_size=11, stride=4, padding=1), nn.ReLU(),
        nn.MaxPool2d(kernel_size=3, stride=2),
        # 减小卷积窗口，使用填充为2来使得输入与输出的高和宽一致，且增大输出通道数
        nn.Conv2d(96, 256, kernel_size=5, padding=2), nn.ReLU(),
        nn.MaxPool2d(kernel_size=3, stride=2),
        # 使用三个连续的卷积层和较小的卷积窗口。
        # 除了最后的卷积层，输出通道的数量进一步增加。
        # 在前两个卷积层之后，汇聚层不用于减少输入的高度和宽度
        nn.Conv2d(256, 384, kernel_size=3, padding=1), nn.ReLU(),
        nn.Conv2d(384, 384, kernel_size=3, padding=1), nn.ReLU(),
        nn.Conv2d(384, 256, kernel_size=3, padding=1), nn.ReLU(),
        nn.MaxPool2d(kernel_size=3, stride=2),
        nn.Flatten(),
        # 这里，全连接层的输出数量是LeNet中的好几倍。使用dropout层来减轻过拟合
        nn.Linear(6400, 4096), nn.ReLU(),
        nn.Dropout(p=0.5),
        nn.Linear(4096, 4096), nn.ReLU(),
        nn.Dropout(p=0.5),
        # 最后是输出层。由于这里使用Fashion-MNIST，所以用类别数为10，而非论文中的1000
        nn.Linear(4096, 10))
    return net

def vgg_block(num_convs, in_channels, out_channels):
    '''
    VGG块：使用了带有3 X 3卷积核（有多个）、填充为1（保持高度和宽度）的卷积层，和带有2 X 2汇聚窗口、步幅为2（每个块后的分辨率减半）的最大汇聚层。
    :param num_convs: 卷积层的数量
    :param in_channels: 输入通道的数量
    :param out_channels: 输出通道的数量
    :return:
    '''
    layers = []
    for _ in range(num_convs):
        layers.append(nn.Conv2d(in_channels, out_channels,
                                kernel_size=3, padding=1))
        layers.append(nn.ReLU())
        in_channels = out_channels
    layers.append(nn.MaxPool2d(kernel_size=2,stride=2))
    return nn.Sequential(*layers)

def vgg(conv_arch):
    '''
    VGG-11
    原始VGG网络有5个卷积块，其中前两个块各有一个卷积层，后三个块各包含两个卷积层。 第一个模块有64个输出通道，
    每个后续模块将输出通道数量翻倍，直到该数字达到512。由于该网络使用8个卷积层和3个全连接层，因此它通常被称为VGG-11。

    LeNet、AlexNet和VGG都有一个共同的设计模式：通过一系列的卷积层与汇聚层来提取空间结构特征；

    给人的启发：可以设计块组合使用

    :param conv_arch: 每个VGG块里卷积层个数和输出通道数，
    如：conv_arch = ((1, 64), (1, 128), (2, 256), (2, 512), (2, 512))
    :return:
    '''
    conv_blks = []
    in_channels = 1
    # 卷积层部分
    for (num_convs, out_channels) in conv_arch:
        conv_blks.append(vgg_block(num_convs, in_channels, out_channels))
        in_channels = out_channels

    return nn.Sequential(
        *conv_blks, nn.Flatten(),
        # 全连接层部分
        nn.Linear(out_channels * 7 * 7, 4096), nn.ReLU(), nn.Dropout(0.5),
        nn.Linear(4096, 4096), nn.ReLU(), nn.Dropout(0.5),
        nn.Linear(4096, 10))


def nin_block(in_channels, out_channels, kernel_size, strides, padding):
    '''
     NiN块以一个普通卷积层开始，后面是两个1 X 1的卷积层。这两个 1 X 1卷积层充当带有ReLU激活函数的逐像素全连接层。
     第一层的卷积窗口形状通常由用户设置。 随后的卷积窗口形状固定为 1 X 1
    :param in_channels: 输入通道
    :param out_channels: 输出通道
    :param kernel_size: 第一层的卷积窗口形状
    :param strides: 第一层的卷积窗口步幅
    :param padding: 第一层的卷积窗口填充
    :return:
    '''
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, strides, padding),
        nn.ReLU(),
        nn.Conv2d(out_channels, out_channels, kernel_size=1), nn.ReLU(),
        nn.Conv2d(out_channels, out_channels, kernel_size=1), nn.ReLU())

def nin():
    '''
     NiN使用窗口形状为 11 X 11、5 X 5、3 X 3的卷积层，输出通道数量与AlexNet中的相同。 每个NiN块后有一个最大汇聚层，
     汇聚窗口形状为3 X 3，步幅为2。

    NiN和AlexNet之间的一个显著区别是NiN去除了容易造成过拟合的全连接层，将它们替换为全局平均汇聚层（即在所有位置上进行求和）。
    该汇聚层通道数量为所需的输出数量

    NiN使用一个NiN块，其输出通道数等于标签类别的数量。最后放一个全局平均汇聚层（global average pooling layer），
    生成一个对数几率 （logits）。NiN设计的一个优点是，它显著减少了模型所需参数的数量。但是，这种设计有时会增加训练模型的时间。
    :return:
    '''
    net = nn.Sequential(
        nin_block(1, 96, kernel_size=11, strides=4, padding=0),
        nn.MaxPool2d(3, stride=2),
        nin_block(96, 256, kernel_size=5, strides=1, padding=2),
        nn.MaxPool2d(3, stride=2),
        nin_block(256, 384, kernel_size=3, strides=1, padding=1),
        nn.MaxPool2d(3, stride=2),
        nn.Dropout(0.5),
        # 标签类别数是10
        nin_block(384, 10, kernel_size=3, strides=1, padding=1),
        nn.AdaptiveAvgPool2d((1, 1)),
        # 将四维的输出转成二维的输出，其形状为(批量大小,10)
        nn.Flatten())
    return net

class Inception(nn.Module):

    '''
    Inception块由四条并行路径组成。 前三条路径使用窗口大小为 1 X 1、3 X 3、5 X 5的卷积层，从不同空间大小中提取信息。
    中间的两条路径在输入上执行1 X 1卷积，以减少通道数，从而降低模型的复杂性。 第四条路径使用3 X 3最大汇聚层，然后使用
    1 X 1卷积层来改变通道数。
    '''

    # c1--c4是每条路径的输出通道数
    def __init__(self, in_channels, c1, c2, c3, c4, **kwargs):
        super(Inception, self).__init__(**kwargs)
        # 线路1，单1x1卷积层
        self.p1_1 = nn.Conv2d(in_channels, c1, kernel_size=1)
        # 线路2，1x1卷积层后接3x3卷积层
        self.p2_1 = nn.Conv2d(in_channels, c2[0], kernel_size=1)
        self.p2_2 = nn.Conv2d(c2[0], c2[1], kernel_size=3, padding=1)
        # 线路3，1x1卷积层后接5x5卷积层
        self.p3_1 = nn.Conv2d(in_channels, c3[0], kernel_size=1)
        self.p3_2 = nn.Conv2d(c3[0], c3[1], kernel_size=5, padding=2)
        # 线路4，3x3最大汇聚层后接1x1卷积层
        self.p4_1 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.p4_2 = nn.Conv2d(in_channels, c4, kernel_size=1)

    def forward(self, x):
        p1 = F.relu(self.p1_1(x))
        p2 = F.relu(self.p2_2(F.relu(self.p2_1(x))))
        p3 = F.relu(self.p3_2(F.relu(self.p3_1(x))))
        p4 = F.relu(self.p4_2(self.p4_1(x)))
        # 在通道维度上连结输出
        return torch.cat((p1, p2, p3, p4), dim=1)

def googlenet():
    # 第一个模块使用64个通道、7 X 7卷积层。
    b1 = nn.Sequential(nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
                       nn.ReLU(),
                       nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
    # 第二个模块使用两个卷积层：第一个卷积层是64个通道、1 X 1卷积层；第二个卷积层使用将通道数量增加三倍的3 X 3卷积层
    b2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1),
                       nn.ReLU(),
                       nn.Conv2d(64, 192, kernel_size=3, padding=1),
                       nn.ReLU(),
                       nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
    # 第三个模块串联两个完整的Inception块。 第一个Inception块的输出通道数为64+128+32+32=256
    # 第二个Inception块的输出通道数增加到 128+182+94+64=480
    b3 = nn.Sequential(Inception(192, 64, (96, 128), (16, 32), 32),
                       Inception(256, 128, (128, 192), (32, 96), 64),
                       nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
    # 第四模块串联了5个Inception块
    b4 = nn.Sequential(Inception(480, 192, (96, 208), (16, 48), 64),
                       Inception(512, 160, (112, 224), (24, 64), 64),
                       Inception(512, 128, (128, 256), (24, 64), 64),
                       Inception(512, 112, (144, 288), (32, 64), 64),
                       Inception(528, 256, (160, 320), (32, 128), 128),
                       nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

    b5 = nn.Sequential(Inception(832, 256, (160, 320), (32, 128), 128),
                       Inception(832, 384, (192, 384), (48, 128), 128),
                       nn.AdaptiveAvgPool2d((1, 1)),
                       nn.Flatten())

    net = nn.Sequential(b1, b2, b3, b4, b5, nn.Linear(1024, 10))
    return net

class BatchNorm(nn.Module):
    '''
    批量规范化层
    '''
    # num_features：完全连接层的输出数量或卷积层的输出通道数。
    # num_dims：2表示完全连接层，4表示卷积层
    def __init__(self, num_features, num_dims):
        super().__init__()
        if num_dims == 2:
            shape = (1, num_features)
        else:
            shape = (1, num_features, 1, 1)
        # 参与求梯度和迭代的拉伸和偏移参数，分别初始化成1和0
        self.gamma = nn.Parameter(torch.ones(shape))
        self.beta = nn.Parameter(torch.zeros(shape))
        # 非模型参数的变量初始化为0和1
        self.moving_mean = torch.zeros(shape)
        self.moving_var = torch.ones(shape)

    def forward(self, X):
        # 如果X不在内存上，将moving_mean和moving_var
        # 复制到X所在显存上
        if self.moving_mean.device != X.device:
            self.moving_mean = self.moving_mean.to(X.device)
            self.moving_var = self.moving_var.to(X.device)
        # 保存更新过的moving_mean和moving_var
        Y, self.moving_mean, self.moving_var = batch_norm(
            X, self.gamma, self.beta, self.moving_mean,
            self.moving_var, eps=1e-5, momentum=0.9)
        return Y

def batch_norm(X, gamma, beta, moving_mean, moving_var, eps, momentum):
    '''
    批量规范化层
    :param X: 输入数据，可以是一个二维张量（对于全连接层）或一个四维张量（对于卷积层）。
    :param gamma:
    :param beta: gamma、beta这两个是可学习的缩放和偏移参数，用于在归一化后对数据进行调整。
    :param moving_mean: 移动平均的均值，用于在推理（或评估）阶段进行归一化。
    :param moving_var: 移动平均的方差，用于在推理（或评估）阶段进行归一化。
    :param eps: 一个小的正数，用于防止分母为零，确保数值稳定性。
    :param momentum: 用于计算移动平均的动量值，通常接近1（如0.9或0.99）。
    :return:
    '''

    # 通过is_grad_enabled来判断当前模式是训练模式还是预测模式
    if not torch.is_grad_enabled():
        # 如果是在预测模式下，直接使用传入的移动平均所得的均值和方差
        X_hat = (X - moving_mean) / torch.sqrt(moving_var + eps)
    else:
        assert len(X.shape) in (2, 4)
        if len(X.shape) == 2:
            # 使用全连接层的情况，计算特征维上的均值和方差
            mean = X.mean(dim=0)
            var = ((X - mean) ** 2).mean(dim=0)
        else:
            # 使用二维卷积层的情况，计算通道维上（axis=1）的均值和方差。
            # 这里我们需要保持X的形状以便后面可以做广播运算
            mean = X.mean(dim=(0, 2, 3), keepdim=True)
            var = ((X - mean) ** 2).mean(dim=(0, 2, 3), keepdim=True)
        # 训练模式下，用当前的均值和方差做标准化
        X_hat = (X - mean) / torch.sqrt(var + eps)
        # 更新移动平均的均值和方差
        moving_mean = momentum * moving_mean + (1.0 - momentum) * mean
        moving_var = momentum * moving_var + (1.0 - momentum) * var
    Y = gamma * X_hat + beta  # 缩放和移位
    return Y, moving_mean.data, moving_var.data

class Residual(nn.Module):  #@save
    '''
    残差块： 残差块里首先有2个有相同输出通道数的 3 X 3卷积层.每个卷积层后接一个批量规范化层和ReLU激活函数。
    '''
    def __init__(self, input_channels, num_channels,
                 use_1x1conv=False, strides=1):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, num_channels,
                               kernel_size=3, padding=1, stride=strides)
        self.conv2 = nn.Conv2d(num_channels, num_channels,
                               kernel_size=3, padding=1)
        if use_1x1conv:
            # use_1x1conv=True时，添加通过 1X1卷积调整通道和分辨率。
            self.conv3 = nn.Conv2d(input_channels, num_channels,
                                   kernel_size=1, stride=strides)
        else:
            # 当use_1x1conv=False时，应用ReLU非线性函数之前，将输入添加到输出
            self.conv3 = None
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.bn2 = nn.BatchNorm2d(num_channels)

    def forward(self, X):
        # 输入先经过卷积层，然后批量规范化，然后是relu激活函数
        Y = F.relu(self.bn1(self.conv1(X)))
        # 在经过卷积层，然后批量规范化
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        Y += X
        return F.relu(Y)

def resnet_block(input_channels, num_channels, num_residuals,
                 first_block=False):
    '''
     ResNet则使用4个由残差块组成的模块,这里是组合残差块
    :param input_channels:
    :param num_channels:
    :param num_residuals:
    :param first_block:
    :return:
    '''
    blk = []
    for i in range(num_residuals):
        if i == 0 and not first_block:
            blk.append(Residual(input_channels, num_channels,
                                use_1x1conv=True, strides=2))
        else:
            blk.append(Residual(num_channels, num_channels))
    return blk

def res_net():
    '''
    残差模型：ResNet的前两层跟之前介绍的GoogLeNet中的一样： 在输出通道数为64、步幅为2的7X7卷积层后，
    接步幅为2的3X3的最大汇聚层。 不同之处在于ResNet每个卷积层后增加了批量规范化层。

    GoogLeNet在后面接了4个由Inception块组成的模块。 ResNet则使用4个由残差块组成的模块，每个模块使用若干个
    同样输出通道数的残差块。
     第一个模块的通道数同输入通道数一致。 由于之前已经使用了步幅为2的最大汇聚层，所以无须减小高和宽。
     之后的每个模块在第一个残差块里将上一个模块的通道数翻倍，并将高和宽减半。
    :return:
    '''
    b1 = nn.Sequential(nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
                       nn.BatchNorm2d(64), nn.ReLU(),
                       nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
    b2 = nn.Sequential(*resnet_block(64, 64, 2, first_block=True))
    b3 = nn.Sequential(*resnet_block(64, 128, 2))
    b4 = nn.Sequential(*resnet_block(128, 256, 2))
    b5 = nn.Sequential(*resnet_block(256, 512, 2))
    net = nn.Sequential(b1, b2, b3, b4, b5,
                        nn.AdaptiveAvgPool2d((1, 1)),
                        nn.Flatten(), nn.Linear(512, 10))
    return net

def conv_block(input_channels, num_channels):
    '''
    卷积块
    :param input_channels:
    :param num_channels:
    :return:
    '''
    return nn.Sequential(
        nn.BatchNorm2d(input_channels), nn.ReLU(),
        nn.Conv2d(input_channels, num_channels, kernel_size=3, padding=1))

class DenseBlock(nn.Module):
    '''
    稠密块:
    一个稠密块由多个卷积块组成，每个卷积块使用相同数量的输出通道。
    在前向传播中，我们将每个卷积块的输入和输出在通道维上连结。
    '''
    def __init__(self, num_convs, input_channels, num_channels):
        super(DenseBlock, self).__init__()
        layer = []
        for i in range(num_convs):
            layer.append(conv_block(
                num_channels * i + input_channels, num_channels))
        self.net = nn.Sequential(*layer)

    def forward(self, X):
        for blk in self.net:
            Y = blk(X)
            # 连接通道维度上每个块的输入和输出
            X = torch.cat((X, Y), dim=1)
        return X

def transition_block(input_channels, num_channels):

    '''
    过渡层：
    通过 1X1 卷积层来减小通道数，并使用步幅为2的平均汇聚层减半高和宽，从而进一步降低模型复杂度。

    由于每个稠密块都会带来通道数的增加，使用过多则会过于复杂化模型。
    :param input_channels:
    :param num_channels:
    :return:
    '''
    return nn.Sequential(
        nn.BatchNorm2d(input_channels), nn.ReLU(),
        nn.Conv2d(input_channels, num_channels, kernel_size=1),
        nn.AvgPool2d(kernel_size=2, stride=2))

def dense_net():
    '''
    稠密模型：
    ResNet和DenseNet的关键区别在于，DenseNet输出是连接（用[,]表示）而不是如ResNet的简单相加
    :return:
    '''
    b1 = nn.Sequential(
        nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
        nn.BatchNorm2d(64), nn.ReLU(),
        nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

    # num_channels为当前的通道数
    num_channels, growth_rate = 64, 32
    # 4个稠密块，每个稠密块使用4个卷积层
    num_convs_in_dense_blocks = [4, 4, 4, 4]
    blks = []
    for i, num_convs in enumerate(num_convs_in_dense_blocks):
        blks.append(DenseBlock(num_convs, num_channels, growth_rate))
        # 上一个稠密块的输出通道数
        num_channels += num_convs * growth_rate
        # 在稠密块之间添加一个转换层，使通道数量减半
        if i != len(num_convs_in_dense_blocks) - 1:
            blks.append(transition_block(num_channels, num_channels // 2))
            num_channels = num_channels // 2

    net = nn.Sequential(
        b1, *blks,
        nn.BatchNorm2d(num_channels), nn.ReLU(),
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten(),
        nn.Linear(num_channels, 10))
    return net