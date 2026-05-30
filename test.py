# test.py

import torch
from generate import load_model_and_tokenizer, generate, gen_acrostic
from config import Config

def userTest():
    print("正在加载微调后的 GPT‑2 诗歌模型...")
    model, tokenizer = load_model_and_tokenizer()
    print("初始化完成！\n")
    while True:
        print("欢迎使用唐诗生成器（GPT‑2 版），\n"
              "输入1 进入首句生成模式\n"
              "输入2 进入藏头诗生成模式\n"
              "输入0 退出\n")
        mode = input().strip()
        if mode == '0':
            break
        elif mode == '1':
            print("请输入您想要的诗歌首句（如“床前明月光”）：")
            start_words = input().strip()
            if not start_words:
                print("输入不能为空")
                continue
            poem = generate(model, tokenizer, start_words, max_new_tokens=Config.max_gen_len)
            print("生成的诗句如下：\n", poem, "\n")
        elif mode == '2':
            print("请输入藏头字串（例如“春江花月夜”）：")
            start_words = input().strip()
            if not start_words:
                print("输入不能为空")
                continue
            poem = gen_acrostic(model, tokenizer, start_words, max_new_tokens=Config.max_gen_len)
            print("生成的藏头诗如下：\n", poem, "\n")
        else:
            print("无效输入，请重试")

if __name__ == '__main__':
    userTest()