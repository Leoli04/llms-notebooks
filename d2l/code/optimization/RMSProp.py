import math
import torch
from d2l import torch as d2l

def init_rmsprop_states(feature_dim):
    s_w = torch.zeros((feature_dim, 1))
    s_b = torch.zeros(1)
    return (s_w, s_b)

def rmsprop(params, states, hyperparams):
    gamma, eps = hyperparams['gamma'], 1e-6
    for p, s in zip(params, states):
        with torch.no_grad():
            s[:] = gamma * s + (1 - gamma) * torch.square(p.grad)
            p[:] -= hyperparams['lr'] * p.grad / torch.sqrt(s + eps)
        p.grad.data.zero_()

if __name__ == '__main__':
    data_iter, feature_dim = d2l.get_data_ch11(batch_size=10)
    d2l.train_ch11(rmsprop, init_rmsprop_states(feature_dim),
                   {'lr': 0.01, 'gamma': 0.9}, data_iter, feature_dim);

    trainer = torch.optim.RMSprop
    d2l.train_concise_ch11(trainer, {'lr': 0.01, 'alpha': 0.9},
                           data_iter)