import math
import os
import random
import torch
from d2l import torch as d2l
from Vocab import count_corpus

def read_ptb():
    """
    将PTB数据集加载到文本行的列表中
    该语料库取自“华尔街日报”的文章，分为训练集、验证集和测试集
    """
    data_dir = d2l.download_extract('ptb')
    # Readthetrainingset.
    with open(os.path.join(data_dir, 'ptb.train.txt')) as f:
        raw_text = f.read()
    return [line.split() for line in raw_text.split('\n')]

#@save
def subsample(sentences, vocab):
    """
    下采样高频词

    下采样（Downsampling）：下采样主要用于减少数据集的大小，以便更快地训练和评估模型，同时减少计算资源的使用。
    它通过减少样本的数量来实现，可以是随机选择样本，也可以是基于某种策略（如保留最具有代表性的样本）来选择样本。

    """
    # 排除未知词元'<unk>'
    sentences = [[token for token in line if vocab[token] != vocab.unk]
                 for line in sentences]
    counter = count_corpus(sentences)
    num_tokens = sum(counter.values())

    # 如果在下采样期间保留词元，则返回True
    def keep(token):

        return(random.uniform(0, 1) <
               #  # 计算一个与词频成反比的概率阈值，词频越高，计算出的阈值越小。
               math.sqrt(1e-4 / counter[token] * num_tokens))

    return ([[token for token in line if keep(token)] for line in sentences],
            counter)

def compare_counts(token):
    return (f'"{token}"的数量：'
            f'之前={sum([l.count(token) for l in sentences])}, '
            f'之后={sum([l.count(token) for l in subsampled])}')


#@save
def get_centers_and_contexts(corpus, max_window_size):
    """返回跳元模型中的中心词和上下文词"""
    centers, contexts = [], []
    for line in corpus:
        # 要形成“中心词-上下文词”对，每个句子至少需要有2个词
        if len(line) < 2:
            continue
        centers += line
        for i in range(len(line)):  # 上下文窗口中间i
            window_size = random.randint(1, max_window_size)
            indices = list(range(max(0, i - window_size),
                                 min(len(line), i + 1 + window_size)))
            # 从上下文词中排除中心词
            indices.remove(i)
            contexts.append([line[idx] for idx in indices])
    return centers, contexts

#@save
class RandomGenerator:
    """根据n个采样权重在{1,...,n}中随机抽取"""
    def __init__(self, sampling_weights):
        # Exclude
        self.population = list(range(1, len(sampling_weights) + 1))
        self.sampling_weights = sampling_weights
        self.candidates = []
        self.i = 0

    def draw(self):
        if self.i == len(self.candidates):
            # 缓存k个随机采样结果
            self.candidates = random.choices(
                self.population, self.sampling_weights, k=10000)
            self.i = 0
        self.i += 1
        return self.candidates[self.i - 1]

#@save
def get_negatives(all_contexts, vocab, counter, K):
    """返回负采样中的噪声词"""
    # 索引为1、2、...（索引0是词表中排除的未知标记）
    sampling_weights = [counter[vocab.to_tokens(i)]**0.75
                        for i in range(1, len(vocab))]
    all_negatives, generator = [], RandomGenerator(sampling_weights)
    for contexts in all_contexts:
        negatives = []
        while len(negatives) < len(contexts) * K:
            neg = generator.draw()
            # 噪声词不能是上下文词
            if neg not in contexts:
                negatives.append(neg)
        all_negatives.append(negatives)
    return all_negatives

#@save
def batchify(data):
    """返回带有负采样的跳元模型的小批量样本"""
    max_len = max(len(c) + len(n) for _, c, n in data)
    centers, contexts_negatives, masks, labels = [], [], [], []
    for center, context, negative in data:
        cur_len = len(context) + len(negative)
        centers += [center]
        contexts_negatives += \
            [context + negative + [0] * (max_len - cur_len)]
        masks += [[1] * cur_len + [0] * (max_len - cur_len)]
        labels += [[1] * len(context) + [0] * (max_len - len(context))]
    return (torch.tensor(centers).reshape((-1, 1)), torch.tensor(
        contexts_negatives), torch.tensor(masks), torch.tensor(labels))

#@save
def load_data_ptb(batch_size, max_window_size, num_noise_words):
    """下载PTB数据集，然后将其加载到内存中"""
    num_workers = d2l.get_dataloader_workers()
    sentences = read_ptb()
    vocab = d2l.Vocab(sentences, min_freq=10)
    subsampled, counter = subsample(sentences, vocab)
    corpus = [vocab[line] for line in subsampled]
    all_centers, all_contexts = get_centers_and_contexts(
        corpus, max_window_size)
    all_negatives = get_negatives(
        all_contexts, vocab, counter, num_noise_words)

    class PTBDataset(torch.utils.data.Dataset):
        def __init__(self, centers, contexts, negatives):
            assert len(centers) == len(contexts) == len(negatives)
            self.centers = centers
            self.contexts = contexts
            self.negatives = negatives

        def __getitem__(self, index):
            return (self.centers[index], self.contexts[index],
                    self.negatives[index])

        def __len__(self):
            return len(self.centers)

    dataset = PTBDataset(all_centers, all_contexts, all_negatives)

    data_iter = torch.utils.data.DataLoader(
        dataset, batch_size, shuffle=True,
        collate_fn=batchify, num_workers=num_workers)
    return data_iter, vocab

if __name__ == '__main__':
    d2l.DATA_HUB['ptb'] = (d2l.DATA_URL + 'ptb.zip',
                           '319d85e578af0cdc590547f26231e4e31cdf1e42')

    sentences = read_ptb()
    # 出现次数少于10次的任何单词都将由“<unk>”词元替换
    vocab = d2l.Vocab(sentences, min_freq=10)
    print(f'vocab size: {len(vocab)}')

    # 下采样
    subsampled, counter = subsample(sentences, vocab)

    d2l.show_list_len_pair_hist(
        ['origin', 'subsampled'], '# tokens per sentence',
        'count', sentences, subsampled);

    print(compare_counts('the'),compare_counts('join'))
    # 们将词元映射到它们在语料库中的索引
    corpus = [vocab[line] for line in subsampled]
    print(corpus[:3])

    tiny_dataset = [list(range(7)), list(range(7, 10))]
    print('数据集', tiny_dataset)
    for center, context in zip(*get_centers_and_contexts(tiny_dataset, 2)):
        print('中心词', center, '的上下文词是', context)

    all_centers, all_contexts = get_centers_and_contexts(corpus, 5)
    print(f'# “中心词-上下文词对”的数量: {sum([len(contexts) for contexts in all_contexts])}')

    # @save
    generator = RandomGenerator([2, 3, 4])
    [generator.draw() for _ in range(10)]

    # 随机抽取了K个（实验中为5个）噪声词
    all_negatives = get_negatives(all_contexts, vocab, counter, 5)

    # 使用一个小批量的两个样本来测试batchify函数
    x_1 = (1, [2, 2], [3, 3, 3, 3])
    x_2 = (1, [2, 2, 2], [3, 3])
    batch = batchify((x_1, x_2))

    names = ['centers', 'contexts_negatives', 'masks', 'labels']
    for name, data in zip(names, batch):
        print(name, '=', data)

    # 打印数据迭代器的第一个小批量
    data_iter, vocab = load_data_ptb(512, 5, 5)
    for batch in data_iter:
        for name, data in zip(names, batch):
            print(name, 'shape:', data.shape)
        break