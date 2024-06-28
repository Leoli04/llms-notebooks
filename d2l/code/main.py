import torch


from train_util import train_ch3
from train_util import train_ch6
from data_util import load_data_fashion_mnist
from gpu_util import try_gpu
from cnn.cnn_model import lenet,alexnet



def lenet_train():
    net = lenet()

    # 输入为28*28单通道
    X = torch.rand(size=(1, 1, 28, 28), dtype=torch.float32)
    for layer in net:
        X = layer(X)
        print(layer.__class__.__name__, 'output shape: \t', X.shape)

    # 模型训练
    batch_size = 256
    train_iter, test_iter = load_data_fashion_mnist(batch_size=batch_size)

    lr, num_epochs = 0.9, 10
    train_ch6(net, train_iter, test_iter, num_epochs, lr, try_gpu())

def alexnet_train():
    net = alexnet()
    X = torch.randn(1, 1, 224, 224)
    for layer in net:
        X = layer(X)
        print(layer.__class__.__name__, 'output shape:\t', X.shape)

    batch_size = 128
    train_iter, test_iter = load_data_fashion_mnist(batch_size, resize=224)

    lr, num_epochs = 0.01, 10
    train_ch6(net, train_iter, test_iter, num_epochs, lr, try_gpu())

if __name__ == '__main__':

    ######################################
    # cnn
    ######################################
    lenet_train()

    alexnet_train