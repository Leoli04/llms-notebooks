## 介绍
PyTorch 是由 Facebook 开发，基于 Torch 开发，从并不常用的 Lua 语言转为 Python 语言开发的深度学习框架，
Torch 是 TensorFlow 开源前非常出名的一个深度学习框架，而 PyTorch 在开源后由于其使用简单，动态计算图的特
性得到非常多的关注，并且成为了 TensorFlow 的 最大竞争对手。

## Pytorch 是什么
Pytorch 是一个基于 Python 的科学计算库，它面向以下两种用途：
- NumPy 的替代品，可利用 GPU 和其他加速器的强大功能。
- 一个自动微分库，可用于实现神经网络。

## 安装
[参考文档](https://pytorch.org/get-started/locally/)

```shell
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

#or

conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
```
> NOTE: 最新的 PyTorch 依赖 Python 3.8 及已上版本.

## 学习文档
[get-started](https://pytorch.org/get-started/locally/)
[github](https://github.com/pytorch/pytorch)
[论坛](https://discuss.pytorch.org/)