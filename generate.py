# generate.py

import torch
import numpy as np
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from config import Config

device = Config.device if hasattr(Config, 'device') else torch.device('cuda' if Config.use_gpu else 'cpu')

# 全局加载模型和 tokenizer（仅在第一次调用时加载）
_model = None
_tokenizer = None

def load_model_and_tokenizer():
    global _model, _tokenizer
    if _model is None:
        model_path = Config.gpt2_output_dir  # 微调后模型保存路径
        _tokenizer = GPT2Tokenizer.from_pretrained(model_path)
        _model = GPT2LMHeadModel.from_pretrained(model_path)
        if _tokenizer.pad_token is None:
            _tokenizer.pad_token = _tokenizer.eos_token
        _model.to(device)
        _model.eval()
    return _model, _tokenizer

def generate(model, tokenizer, prompt, max_new_tokens=Config.max_gen_len, temperature=0.7, top_k=50, top_p=0.95):
    """普通诗歌生成，prompt 为字符串（如首句或几个字）"""
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.2
        )
    generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # 去掉可能重复的 prompt 部分
    if generated.startswith(prompt):
        generated = generated[len(prompt):]
    return generated

def gen_acrostic(model, tokenizer, start_words, prefix_words=None, max_new_tokens=Config.max_gen_len):
    """
    生成藏头诗。
    start_words: 藏头字串，例如 "春江花月"
    prefix_words: 可选前缀，如 "作者：李白\n标题：望月\n\n"
    """
    if prefix_words is None:
        prefix_words = ""
    result = list(start_words)
    full_text = prefix_words + start_words
    # 逐字强制生成
    for i, target_char in enumerate(start_words):
        # 当前已生成的部分作为 prompt
        prompt = full_text
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        # 强制下一个 token 必须是 target_char
        target_id = tokenizer.encode(target_char, add_special_tokens=False)[0]
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits[0, -1, :]      # 最后一个位置的 logits
            # 将目标 token 的 logits 设为极大值，其他设为极小
            new_logits = torch.full_like(logits, -float('Inf'))
            new_logits[target_id] = logits[target_id]
            next_token = torch.argmax(new_logits).unsqueeze(0).unsqueeze(0)
            # 生成下一个 token
            generated_ids = torch.cat([inputs['input_ids'], next_token], dim=-1)
            # 继续生成剩余内容（不强制后续字）
            if i == len(start_words) - 1:
                # 最后一个字后自由生成
                output = model.generate(
                    generated_ids,
                    max_new_tokens=max_new_tokens - len(generated_ids[0]),
                    temperature=0.7,
                    top_k=50,
                    top_p=0.95,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id
                )
            else:
                # 中间字只需生成刚强制加入的字，然后循环
                output = generated_ids
        # 更新 full_text
        full_text = tokenizer.decode(output[0], skip_special_tokens=True)
    # 最终完整文本
    if full_text.startswith(prefix_words):
        full_text = full_text[len(prefix_words):]
    return full_text

# 兼容旧接口（用于 test.py 等）
def generate_wrapper(model, start_words, ix2word, word2ix, prefix_words=None, use_transformer=False, max_gen_len=None):
    """包装函数，保持与原 generate 相同的调用签名"""
    global _model, _tokenizer
    if _model is None:
        _model, _tokenizer = load_model_and_tokenizer()
    max_len = max_gen_len if max_gen_len else Config.max_gen_len
    return generate(_model, _tokenizer, start_words, max_new_tokens=max_len)

def gen_acrostic_wrapper(model, start_words, ix2word, word2ix, prefix_words=None, use_transformer=False):
    global _model, _tokenizer
    if _model is None:
        _model, _tokenizer = load_model_and_tokenizer()
    return gen_acrostic(_model, _tokenizer, start_words, prefix_words=prefix_words)