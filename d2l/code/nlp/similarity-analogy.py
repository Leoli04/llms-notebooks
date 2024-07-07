import os
import torch
from torch import nn
from d2l import torch as d2l

#@save
class TokenEmbedding:
    """GloVe嵌入"""
    def __init__(self, embedding_name):
        self.idx_to_token, self.idx_to_vec = self._load_embedding(
            embedding_name)
        self.unknown_idx = 0
        self.token_to_idx = {token: idx for idx, token in
                             enumerate(self.idx_to_token)}

    def _load_embedding(self, embedding_name):
        idx_to_token, idx_to_vec = ['<unk>'], []
        data_dir = d2l.download_extract(embedding_name)
        # GloVe网站：https://nlp.stanford.edu/projects/glove/
        # fastText网站：https://fasttext.cc/
        with open(os.path.join(data_dir, 'vec.txt'), 'r',errors='ignore') as f:
            for line in f:
                elems = line.rstrip().split(' ')
                token, elems = elems[0], [float(elem) for elem in elems[1:]]
                # 跳过标题信息，例如fastText中的首行
                if len(elems) > 1:
                    idx_to_token.append(token)
                    idx_to_vec.append(elems)
        idx_to_vec = [[0] * len(idx_to_vec[0])] + idx_to_vec
        return idx_to_token, torch.tensor(idx_to_vec)

    def __getitem__(self, tokens):
        indices = [self.token_to_idx.get(token, self.unknown_idx)
                   for token in tokens]
        vecs = self.idx_to_vec[torch.tensor(indices)]
        return vecs

    def __len__(self):
        return len(self.idx_to_token)

def knn(W, x, k):
    '''
    使用余弦相似度 在w中与x相似的查找top k词
    余弦相似度是一种通过计算两个向量在向量空间中夹角的余弦值来评估它们相似度的方法。
    在文本处理领域，词可以被表示为向量，这些向量通常基于词频、TF-IDF值或其他词嵌入技术（如Word2Vec、GloVe等）构建。因此，通过计算这些词向量之间的余弦相似度，我们可以评估词之间的相似程度。
    :param W: 词表的向量
    :param x: 查询词向量
    :param k:
    :return:
    '''
    # 增加1e-9以获得数值稳定性
    cos = torch.mv(W, x.reshape(-1,)) / (
        torch.sqrt(torch.sum(W * W, axis=1) + 1e-9) *
        torch.sqrt((x * x).sum()))
    _, topk = torch.topk(cos, k=k)
    return topk, [cos[int(i)] for i in topk]

def get_similar_tokens(query_token, k, embed):
    '''
    词的相似度
    :param query_token: 查询词
    :param k: 相似度个数
    :param embed: 词表
    :return:
    '''
    topk, cos = knn(embed.idx_to_vec, embed[[query_token]], k + 1)
    # 返回的第一个词是查询词本身
    for i, c in zip(topk[1:], cos[1:]):  # 排除输入词
        print(f'{embed.idx_to_token[int(i)]}：cosine相似度={float(c):.3f}')

def get_analogy(token_a, token_b, token_c, embed):
    '''
    词类比
    :param token_a:
    :param token_b:
    :param token_c:
    :param embed:
    :return:
    '''
    vecs = embed[[token_a, token_b, token_c]]
    x = vecs[1] - vecs[0] + vecs[2]
    topk, cos = knn(embed.idx_to_vec, x, 1)
    return embed.idx_to_token[int(topk[0])]  # 删除未知词


if __name__ == '__main__':
    ############################
    # 使用glove.6b.50d 中预先练的词向量 处理词的相似度和类比任务
    #############################
    # @save
    d2l.DATA_HUB['glove.6b.50d'] = (d2l.DATA_URL + 'glove.6B.50d.zip',
                                    '0b8703943ccdb6eb788e6f091b8946e82231bc4d')

    # @save
    d2l.DATA_HUB['glove.6b.100d'] = (d2l.DATA_URL + 'glove.6B.100d.zip',
                                     'cd43bfb07e44e6f27cbcc7bc9ae3d80284fdaf5a')

    # @save
    d2l.DATA_HUB['glove.42b.300d'] = (d2l.DATA_URL + 'glove.42B.300d.zip',
                                      'b5116e234e9eb9076672cfeabf5469f3eec904fa')

    # @save
    d2l.DATA_HUB['wiki.en'] = (d2l.DATA_URL + 'wiki.en.zip',
                               'c1816da3821ae9f43899be655002f6c723e91b88')

    glove_6b50d = TokenEmbedding('glove.6b.50d')

    print(len(glove_6b50d))
    # 查看数据
    print(glove_6b50d.token_to_idx['beautiful'], glove_6b50d.idx_to_token[3367])
    # 在词表中找到与“chip”一词语义最相似的三个词
    print(get_similar_tokens('chip', 3, glove_6b50d))
    print(get_similar_tokens('baby', 3, glove_6b50d))
    print(get_similar_tokens('beautiful', 3, glove_6b50d))

    print(get_analogy('man', 'woman', 'son', glove_6b50d))
    print(get_analogy('bad', 'worst', 'big', glove_6b50d))