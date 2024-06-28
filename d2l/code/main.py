import torch
from torch import nn

from train_util import train_ch3
from train_util import train_ch6
from data_util import load_data_fashion_mnist
from gpu_util import try_gpu
from cnn.cnn_model import lenet,alexnet,vgg,nin,googlenet,BatchNorm,res_net


def input_shape_change(X,net):
    '''
    查看输入经过模型的每层的变化
    :param X: 输入
    :param net: 模型
    :return:
    '''
    for layer in net:
        X = layer(X)
        print(layer.__class__.__name__, 'output shape: \t', X.shape)

def train_fashion_mnist(lr,num_epochs,batch_size,net):

    train_iter, test_iter = load_data_fashion_mnist(batch_size, resize=224)
    train_ch6(net, train_iter, test_iter, num_epochs, lr, try_gpu())

def lenet_train():
    net = lenet()

    # 输入为28*28单通道
    X = torch.rand(size=(1, 1, 28, 28), dtype=torch.float32)

    input_shape_change(X,net)

    # 模型训练
    batch_size = 256

    lr, num_epochs = 0.9, 10

    train_fashion_mnist(lr,num_epochs,batch_size,net)


def alexnet_train():
    net = alexnet()
    X = torch.randn(1, 1, 224, 224)

    input_shape_change(X, net)

    batch_size = 128
    lr, num_epochs = 0.01, 10
    train_fashion_mnist(lr, num_epochs, batch_size, net)


def vgg_train():
    conv_arch = ((1, 64), (1, 128), (2, 256), (2, 512), (2, 512))
    net = vgg(conv_arch)

    X = torch.randn(size=(1, 1, 224, 224))

    input_shape_change(X,net)

    # VGG-11比AlexNet计算量更大，这里使用通道数较少的网络
    ratio = 4
    small_conv_arch = [(pair[0], pair[1] // ratio) for pair in conv_arch]
    net = vgg(small_conv_arch)
    lr, num_epochs, batch_size = 0.05, 10, 128
    train_fashion_mnist(lr, num_epochs, batch_size, net)

def nin_train():
    net = nin()
    X = torch.rand(size=(1, 1, 224, 224))
    input_shape_change(X,net)
    # 训练模型
    lr, num_epochs, batch_size = 0.1, 10, 128
    train_fashion_mnist(lr, num_epochs, batch_size, net)


def googlenet_train():
   net = googlenet()
   X = torch.rand(size=(1, 1, 96, 96))
   input_shape_change(X,net)

   lr, num_epochs, batch_size = 0.1, 10, 128
   train_fashion_mnist(lr, num_epochs, batch_size, net)

def batch_norm_train():
    # net = nn.Sequential(
    #     nn.Conv2d(1, 6, kernel_size=5), BatchNorm(6, num_dims=4), nn.Sigmoid(),
    #     nn.AvgPool2d(kernel_size=2, stride=2),
    #     nn.Conv2d(6, 16, kernel_size=5), BatchNorm(16, num_dims=4), nn.Sigmoid(),
    #     nn.AvgPool2d(kernel_size=2, stride=2), nn.Flatten(),
    #     nn.Linear(16 * 4 * 4, 120), BatchNorm(120, num_dims=2), nn.Sigmoid(),
    #     nn.Linear(120, 84), BatchNorm(84, num_dims=2), nn.Sigmoid(),
    #     nn.Linear(84, 10))
    net = nn.Sequential(
        nn.Conv2d(1, 6, kernel_size=5), nn.BatchNorm2d(6), nn.Sigmoid(),
        nn.AvgPool2d(kernel_size=2, stride=2),
        nn.Conv2d(6, 16, kernel_size=5), nn.BatchNorm2d(16), nn.Sigmoid(),
        nn.AvgPool2d(kernel_size=2, stride=2), nn.Flatten(),
        nn.Linear(256, 120), nn.BatchNorm1d(120), nn.Sigmoid(),
        nn.Linear(120, 84), nn.BatchNorm1d(84), nn.Sigmoid(),
        nn.Linear(84, 10))
    lr, num_epochs, batch_size = 1.0, 10, 256
    train_fashion_mnist(lr, num_epochs, batch_size, net)

def res_net_train():
    net = res_net()
    X = torch.rand(size=(1, 1, 224, 224))

    input_shape_change(X, net)

    lr, num_epochs, batch_size = 0.05, 10, 256

    train_fashion_mnist(lr, num_epochs, batch_size, net)


if __name__ == '__main__':

    ######################################
    # cnn
    ######################################
    # lenet_train()

    # alexnet_train()

    # googlenet_train()
    # batch_norm_train()

    res_net_train()