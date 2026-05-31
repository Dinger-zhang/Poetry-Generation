import os

import numpy as np
import torch as t
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from torchnet import meter
from tqdm import tqdm

from config import Config
from generate import generate
from model import PoetryModel


class PoetryDataset(Dataset):
    def __init__(self, data, pad_index):
        self.samples = []
        for row in data:
            sample = row
            if pad_index is not None:
                non_pad = sample != pad_index
                if non_pad.any():
                    sample = sample[np.argmax(non_pad) :]
            if len(sample) > 1:
                self.samples.append(t.from_numpy(sample).long())

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return self.samples[index]


class PadCollate(object):
    def __init__(self, pad_index):
        self.pad_index = pad_index

    def __call__(self, batch):
        max_len = max(item.size(0) for item in batch)
        fill_value = 0 if self.pad_index is None else self.pad_index
        padded = batch[0].new_full((len(batch), max_len), fill_value)
        for index, item in enumerate(batch):
            padded[index, : item.size(0)] = item
        return padded


def _make_collate_fn(pad_index):
    return PadCollate(pad_index)


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
    pad_index = word2ix.get("</s>")
    dataset = PoetryDataset(data, pad_index)
    dataloader = DataLoader(
        dataset,
        batch_size=Config.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=device.type == "cuda",
        collate_fn=_make_collate_fn(pad_index),
    )

    model = PoetryModel(
        len(word2ix),
        embedding_dim=Config.embedding_dim,
        hidden_dim=Config.hidden_dim,
        padding_idx=pad_index,
    )
    optimizer = optim.AdamW(model.parameters(), lr=Config.lr, weight_decay=Config.weight_decay)
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
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
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
