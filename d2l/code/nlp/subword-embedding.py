import collections

def get_max_freq_pair(token_freqs):
    '''
    :param token_freqs: 词的频率字典
    :return: 返回词内最频繁的连续符号对，其中词来自输入词典token_freqs的键。
    '''
    pairs = collections.defaultdict(int)
    for token, freq in token_freqs.items():
        symbols = token.split()
        for i in range(len(symbols) - 1):
            # “pairs”的键是两个连续符号的元组
            pairs[symbols[i], symbols[i + 1]] += freq
    return max(pairs, key=pairs.get)  # 具有最大值的“pairs”键


def merge_symbols(max_freq_pair, token_freqs, symbols):
    '''
    合并最频繁的连续符号对以产生新符号
    :param max_freq_pair: 词内最频繁的连续符号对
    :param token_freqs: 词的频率
    :param symbols: 词表
    :return:
    '''
    symbols.append(''.join(max_freq_pair))
    new_token_freqs = dict()
    for token, freq in token_freqs.items():
        new_token = token.replace(' '.join(max_freq_pair),
                                  ''.join(max_freq_pair))
        new_token_freqs[new_token] = token_freqs[token]
    return new_token_freqs

def segment_BPE(tokens, symbols):
    outputs = []
    for token in tokens:
        start, end = 0, len(token)
        cur_output = []
        # 具有符号中可能最长子字的词元段
        while start < len(token) and start < end:
            if token[start: end] in symbols:
                cur_output.append(token[start: end])
                start = end
                end = len(token)
            else:
                end -= 1
        if start < len(token):
            cur_output.append('[UNK]')
        outputs.append(' '.join(cur_output))
    return outputs


if __name__ == '__main__':

    # 将符号词表初始化为所有英文小写字符、特殊的词尾符号'_'和特殊的未知符号'[UNK]'

    symbols = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm',
               'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z',
               '_', '[UNK]']
    # 映射数据集中的词、频率
    raw_token_freqs = {'fast_': 4, 'faster_': 3, 'tall_': 5, 'taller_': 4}
    token_freqs = {}
    for token, freq in raw_token_freqs.items():
        token_freqs[' '.join(list(token))] = raw_token_freqs[token]
    print(token_freqs)

    num_merges = 10
    for i in range(num_merges):
        max_freq_pair = get_max_freq_pair(token_freqs)
        token_freqs = merge_symbols(max_freq_pair, token_freqs, symbols)
        print(f'合并# {i + 1}:', max_freq_pair)

    print(symbols)

    print(list(token_freqs.keys()))

    tokens = ['tallest_', 'fatter_']
    print(segment_BPE(tokens, symbols))

