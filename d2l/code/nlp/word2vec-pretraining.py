import math
import torch
from torch import nn
from d2l import torch as d2l


def skip_gram(center, contexts_and_negatives, embed_v, embed_u):
    v = embed_v(center)
    u = embed_u(contexts_and_negatives)
    pred = torch.bmm(v, u.permute(0, 2, 1))
    return pred

if __name__ == '__main__':
    batch_size, max_window_size, num_noise_words = 512, 5, 5
    data_iter, vocab = d2l.load_data_ptb(batch_size, max_window_size,
                                         num_noise_words)