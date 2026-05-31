import os

import numpy as np
import torch as t

from config import Config
from generate import gen_acrostic, generate
from model import PoetryModel


def _device():
    if Config.use_gpu and t.cuda.is_available():
        return t.device("cuda")
    return t.device("cpu")


def userTest():
    print("正在初始化......")
    device = _device()
    datas = np.load(Config.pickle_path, allow_pickle=True)
    ix2word = datas["ix2word"].item()
    word2ix = datas["word2ix"].item()
    pad_index = word2ix.get("</s>")

    model = PoetryModel(len(ix2word), Config.embedding_dim, Config.hidden_dim, padding_idx=pad_index)
    if not Config.model_path or not os.path.exists(Config.model_path):
        raise FileNotFoundError("请先在config.py中把model_path设置为已训练好的模型文件")
    model.load_state_dict(t.load(Config.model_path, map_location="cpu"))
    model.to(device)
    model.eval()

    print("初始化完成！\n")
    while True:
        print(
            "欢迎使用唐诗生成器，\n"
            "输入1 进入首句生成模式\n"
            "输入2 进入藏头诗生成模式\n"
        )
        mode = int(input())
        if mode == 1:
            print("请输入您想要的诗歌首句，可以是五言或七言")
            start_words = str(input())
            gen_poetry = "".join(generate(model, start_words, ix2word, word2ix))
            print("生成的诗句如下：%s\n" % gen_poetry)
        elif mode == 2:
            print("请输入您想要的诗歌藏头部分，不超过16个字，最好是偶数")
            start_words = str(input())
            gen_poetry = "".join(gen_acrostic(model, start_words, ix2word, word2ix))
            print("生成的诗句如下：%s\n" % gen_poetry)


if __name__ == "__main__":
    userTest()
