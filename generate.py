import torch as t
import numpy as np
from config import Config

device = Config.device if hasattr(Config, 'device') else t.device('cuda' if Config.use_gpu else 'cpu')

def generate(model, start_words, ix2word, word2ix, prefix_words=None, use_transformer=False, max_gen_len=None):
    if max_gen_len is None:
        max_gen_len = Config.max_gen_len

    # 确保生成的序列总长度不超过 transformer_max_len（避免位置编码越界）
    if use_transformer:
        max_total_len = getattr(Config, 'transformer_max_len', 512) - 5   # 留余量
        max_gen_len = min(max_gen_len, max_total_len - len(start_words))

    model.eval()
    results = list(start_words)

    start_id = word2ix['<START>']
    input_ids = [start_id]

    if prefix_words:
        for w in prefix_words:
            if w in word2ix:
                input_ids.append(word2ix[w])

    for w in start_words:
        if w in word2ix:
            input_ids.append(word2ix[w])
        else:
            input_ids.append(word2ix.get('<UNK>', start_id))

    for _ in range(max_gen_len):
        inp = t.tensor(input_ids, dtype=t.long).view(-1, 1).to(device)
        with t.no_grad():
            if use_transformer:
                logits, _ = model(inp)
            else:
                logits, _ = model(inp, hidden=None)
        next_logits = logits[-1, 0, :]
        next_id = t.argmax(next_logits).item()
        next_word = ix2word[next_id]
        if next_word == '<EOP>':
            break
        results.append(next_word)
        input_ids.append(next_id)
        # 防止超长
        if len(input_ids) >= max_total_len - 2:
            break

    return results

def gen_acrostic(model, start_words, ix2word, word2ix, prefix_words=None, use_transformer=False):
    result = []
    start_words_len = len(start_words)
    input_ids = [word2ix['<START>']]

    if prefix_words:
        for w in prefix_words:
            if w in word2ix:
                input_ids.append(word2ix[w])

    pre_word = '<START>'
    index = 0

    max_total_len = getattr(Config, 'transformer_max_len', 512) if use_transformer else 200

    for _ in range(Config.max_gen_len):
        inp = t.tensor(input_ids, dtype=t.long).view(-1, 1).to(device)
        with t.no_grad():
            if use_transformer:
                logits, _ = model(inp)
            else:
                logits, _ = model(inp, hidden=None)
        next_logits = logits[-1, 0, :]
        next_id = t.argmax(next_logits).item()
        next_word = ix2word[next_id]

        if pre_word in {'。', '，', '？', '！', '、', '；', '：', '<START>'}:
            if index < start_words_len:
                w = start_words[index]
                index += 1
                input_ids.append(word2ix.get(w, word2ix['<START>']))
                result.append(w)
                pre_word = w
                continue

        if next_word == '<EOP>':
            break
        result.append(next_word)
        input_ids.append(next_id)
        pre_word = next_word
        if len(input_ids) >= max_total_len - 2:
            break

    return result