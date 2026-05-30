import os

import numpy as np
import torch as t
from torch import nn, optim
from torch.utils.data import DataLoader
from torchnet import meter
from tqdm import tqdm

from config import Config
from generate import generate
from model import PoetryModel


def _device():
    if Config.use_gpu and t.cuda.is_available():
        return t.device("cuda")
    return t.device("cpu")


def train():
    Config.device = _device()
    device = Config.device

    datas = np.load(Config.pickle_path, allow_pickle=True)
    data = datas["data"]
    ix2word = datas["ix2word"].item()
    word2ix = datas["word2ix"].item()
    data = t.from_numpy(data)
    dataloader = DataLoader(
        data,
        batch_size=Config.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=device.type == "cuda",
    )

    model = PoetryModel(
        len(word2ix),
        embedding_dim=Config.embedding_dim,
        hidden_dim=Config.hidden_dim,
    )
    optimizer = optim.Adam(model.parameters(), lr=Config.lr, weight_decay=Config.weight_decay)
    pad_index = word2ix.get("</s>")
    criterion = nn.CrossEntropyLoss(ignore_index=pad_index) if pad_index is not None else nn.CrossEntropyLoss()

    if Config.model_path:
        if not os.path.exists(Config.model_path):
            raise FileNotFoundError("模型文件不存在: %s" % Config.model_path)
        model.load_state_dict(t.load(Config.model_path, map_location="cpu"))

    model.to(device)
    loss_meter = meter.AverageValueMeter()

    with open(Config.result_path, "a", encoding="utf-8") as f:
        f.write("\n===== Transformer training run =====\n")
        for epoch in range(Config.epoch):
            loss_meter.reset()
            model.train()
            for _, data_ in tqdm(enumerate(dataloader), total=len(dataloader)):
                data_ = data_.long().transpose(1, 0).contiguous().to(device)
                optimizer.zero_grad()

                input_, target = data_[:-1, :], data_[1:, :]
                output, _ = model(input_)
                loss = criterion(output, target.contiguous().view(-1))
                loss.backward()
                optimizer.step()
                loss_meter.add(loss.item())

            message = "Epoch %s 训练损失为 %.6f" % (epoch + 1, loss_meter.mean)
            print(message)
            f.write(message + "\n")

            for word in list("春江花朝秋月夜"):
                gen_poetry = "".join(generate(model, word, ix2word, word2ix))
                print(gen_poetry)
                f.write(gen_poetry + "\n\n")
                f.flush()

            t.save(model.state_dict(), "%s_%s.pth" % (Config.model_prefix, epoch + 1))


if __name__ == "__main__":
    train()
