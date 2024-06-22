
import torch
from torch import nn
from d2l import torch as d2l

from RNNModel import RNNModel
from RNNModelScratch import RNNModelScratch

from train_handle import train_ch8
from data_handle import load_data_time_machine


def get_lstm_params(vocab_size, num_hiddens, device):
    '''
    初始化模型参数
    '''

    num_inputs = num_outputs = vocab_size

    def normal(shape):
        return torch.randn(size=shape, device=device)*0.01

    def three():
        return (normal((num_inputs, num_hiddens)),
                normal((num_hiddens, num_hiddens)),
                torch.zeros(num_hiddens, device=device))

    W_xi, W_hi, b_i = three()  # 输入门参数
    W_xf, W_hf, b_f = three()  # 遗忘门参数
    W_xo, W_ho, b_o = three()  # 输出门参数
    W_xc, W_hc, b_c = three()  # 候选记忆元参数
    # 输出层参数
    W_hq = normal((num_hiddens, num_outputs))
    b_q = torch.zeros(num_outputs, device=device)
    # 附加梯度
    params = [W_xi, W_hi, b_i, W_xf, W_hf, b_f, W_xo, W_ho, b_o, W_xc, W_hc,
              b_c, W_hq, b_q]
    for param in params:
        param.requires_grad_(True)
    return params

def init_lstm_state(batch_size, num_hiddens, device):
    '''
    初始化lstm隐藏状态
    长短期记忆网络的隐状态需要返回一个额外的记忆元， 单元的值为0，形状为（批量大小，隐藏单元数）
    :param batch_size:
    :param num_hiddens:
    :param device:
    :return:
    '''

    return (torch.zeros((batch_size, num_hiddens), device=device),
            torch.zeros((batch_size, num_hiddens), device=device))

def lstm(inputs, state, params):
    '''
    模型定义
    提供三个门和一个额外的记忆元。 请注意，只有隐状态才会传递到输出层，而记忆元C不直接参与输出计算。
    :param inputs:
    :param state:
    :param params:
    :return:
    '''
    [W_xi, W_hi, b_i, W_xf, W_hf, b_f, W_xo, W_ho, b_o, W_xc, W_hc, b_c,
     W_hq, b_q] = params
    (H, C) = state
    outputs = []
    for X in inputs:
        I = torch.sigmoid((X @ W_xi) + (H @ W_hi) + b_i)
        F = torch.sigmoid((X @ W_xf) + (H @ W_hf) + b_f)
        O = torch.sigmoid((X @ W_xo) + (H @ W_ho) + b_o)
        C_tilda = torch.tanh((X @ W_xc) + (H @ W_hc) + b_c)
        C = F * C + I * C_tilda
        H = O * torch.tanh(C)
        Y = (H @ W_hq) + b_q
        outputs.append(Y)
    return torch.cat(outputs, dim=0), (H, C)

def lstm_train(type='self'):
    # 加载数据
    batch_size, num_steps = 32, 35
    train_iter, vocab = load_data_time_machine(batch_size, num_steps)

    # 超参数
    vocab_size, num_hiddens, device = len(vocab), 256, d2l.try_gpu()
    num_epochs, lr = 500, 1

    # 训练
    if(type=='self'):
        model = RNNModelScratch(len(vocab), num_hiddens, device, get_lstm_params,
                                    init_lstm_state, lstm)
        train_ch8(model, train_iter, vocab, lr, num_epochs, device)

    else:
        # 通过pytorch实现
        num_inputs = vocab_size
        lstm_layer = nn.LSTM(num_inputs, num_hiddens)
        model = RNNModel(lstm_layer, len(vocab))
        model = model.to(device)
        train_ch8(model, train_iter, vocab, lr, num_epochs, device)

if __name__ == '__main__':
    lstm_train('self')

    lstm_train('pytorch')
