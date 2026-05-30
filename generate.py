import torch as t

from config import Config


def _model_device(model):
    return next(model.parameters()).device


def _token_tensor(token, word2ix, device):
    if token not in word2ix:
        raise KeyError("词表中不存在字符: %s" % token)
    return t.tensor([word2ix[token]], device=device).view(1, 1).long()


def _next_token(output, ix2word, word2ix):
    logits = output[0].detach().clone()
    for token in ("</s>", "<START>"):
        index = word2ix.get(token)
        if index is not None:
            logits[index] = float("-inf")
    top_index = logits.topk(1)[1][0].item()
    return top_index, ix2word[top_index]


# 给定首句生成诗歌
def generate(model, start_words, ix2word, word2ix, prefix_words=None):
    results = list(start_words)
    start_words_len = len(start_words)
    device = _model_device(model)
    input = _token_tensor("<START>", word2ix, device)
    hidden = None
    was_training = model.training
    model.eval()

    try:
        with t.no_grad():
            if prefix_words:
                for word in prefix_words:
                    output, hidden = model(input, hidden)
                    input = _token_tensor(word, word2ix, device)

            for i in range(Config.max_gen_len):
                output, hidden = model(input, hidden)
                if i < start_words_len:
                    w = results[i]
                    input = _token_tensor(w, word2ix, device)
                else:
                    top_index, w = _next_token(output, ix2word, word2ix)
                    results.append(w)
                    input = t.tensor([top_index], device=device).view(1, 1).long()

                if w == "<EOP>":
                    del results[-1]
                    break
    finally:
        if was_training:
            model.train()

    return results


# 生成藏头诗
def gen_acrostic(model, start_words, ix2word, word2ix, prefix_words=None):
    result = []
    start_words_len = len(start_words)
    device = _model_device(model)
    input = _token_tensor("<START>", word2ix, device)
    index = 0
    pre_word = "<START>"
    hidden = None
    was_training = model.training
    model.eval()

    try:
        with t.no_grad():
            if prefix_words:
                for word in prefix_words:
                    output, hidden = model(input, hidden)
                    input = _token_tensor(word, word2ix, device)

            for i in range(Config.max_gen_len):
                output, hidden = model(input, hidden)
                top_index, w = _next_token(output, ix2word, word2ix)
                if pre_word in {"。", "，", "？", "！", "?", "!", "<START>"}:
                    if index == start_words_len:
                        break
                    w = start_words[index]
                    index += 1
                    input = _token_tensor(w, word2ix, device)
                else:
                    input = t.tensor([top_index], device=device).view(1, 1).long()

                if w == "<EOP>":
                    break
                result.append(w)
                pre_word = w
    finally:
        if was_training:
            model.train()

    return result
