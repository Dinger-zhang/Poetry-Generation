import torch as t

from config import Config


def _model_device(model):
    return next(model.parameters()).device


def _token_index(token, word2ix):
    if token not in word2ix:
        raise KeyError("词表中不存在字符: %s" % token)
    return word2ix[token]


def _token_tensor(token, word2ix, device):
    return t.tensor([_token_index(token, word2ix)], device=device).view(1, 1).long()


def _apply_repetition_penalty(logits, recent_tokens):
    penalty = getattr(Config, "repetition_penalty", 1.0)
    if penalty <= 1.0 or not recent_tokens:
        return logits

    window = getattr(Config, "repetition_window", 0)
    tokens = recent_tokens[-window:] if window else recent_tokens
    for index in set(tokens):
        if logits[index] < 0:
            logits[index] *= penalty
        else:
            logits[index] /= penalty
    return logits


def _filter_logits(logits):
    temperature = max(getattr(Config, "gen_temperature", 1.0), 1e-6)
    logits = logits / temperature

    top_k = int(getattr(Config, "gen_top_k", 0) or 0)
    if top_k > 0 and top_k < logits.size(0):
        threshold = t.topk(logits, top_k)[0][-1]
        logits[logits < threshold] = float("-inf")

    top_p = float(getattr(Config, "gen_top_p", 1.0) or 1.0)
    if 0 < top_p < 1.0:
        sorted_logits, sorted_indices = t.sort(logits, descending=True)
        cumulative_probs = t.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
        sorted_indices_to_remove[0] = False
        logits[sorted_indices[sorted_indices_to_remove]] = float("-inf")

    return logits


def _next_token(output, ix2word, word2ix, recent_tokens=None):
    logits = output[-1].detach().clone()
    for token in ("</s>", "<START>"):
        index = word2ix.get(token)
        if index is not None:
            logits[index] = float("-inf")
    logits = _apply_repetition_penalty(logits, recent_tokens)

    if getattr(Config, "gen_sample", False):
        logits = _filter_logits(logits)
        probs = t.softmax(logits, dim=-1)
        if t.isfinite(probs).all() and probs.sum() > 0:
            top_index = t.multinomial(probs, 1)[0].item()
        else:
            top_index = logits.topk(1)[1][0].item()
    else:
        top_index = logits.topk(1)[1][0].item()
    return top_index, ix2word[top_index]


# 给定首句生成诗歌
def generate(model, start_words, ix2word, word2ix, prefix_words=None):
    results = list(start_words)
    start_words_len = len(start_words)
    device = _model_device(model)
    input = _token_tensor("<START>", word2ix, device)
    recent_tokens = [_token_index("<START>", word2ix)]
    hidden = None
    was_training = model.training
    model.eval()

    try:
        with t.no_grad():
            if prefix_words:
                for word in prefix_words:
                    output, hidden = model(input, hidden)
                    input = _token_tensor(word, word2ix, device)
                    recent_tokens.append(_token_index(word, word2ix))

            for i in range(Config.max_gen_len):
                output, hidden = model(input, hidden)
                if i < start_words_len:
                    w = results[i]
                    input = _token_tensor(w, word2ix, device)
                    recent_tokens.append(_token_index(w, word2ix))
                else:
                    top_index, w = _next_token(output, ix2word, word2ix, recent_tokens)
                    results.append(w)
                    input = t.tensor([top_index], device=device).view(1, 1).long()
                    recent_tokens.append(top_index)

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
    recent_tokens = [_token_index("<START>", word2ix)]
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
                    recent_tokens.append(_token_index(word, word2ix))

            for i in range(Config.max_gen_len):
                output, hidden = model(input, hidden)
                top_index, w = _next_token(output, ix2word, word2ix, recent_tokens)
                if pre_word in {"。", "，", "？", "！", "?", "!", "<START>"}:
                    if index == start_words_len:
                        break
                    w = start_words[index]
                    index += 1
                    input = _token_tensor(w, word2ix, device)
                    recent_tokens.append(_token_index(w, word2ix))
                else:
                    input = t.tensor([top_index], device=device).view(1, 1).long()
                    recent_tokens.append(top_index)

                if w == "<EOP>":
                    break
                result.append(w)
                pre_word = w
    finally:
        if was_training:
            model.train()

    return result
