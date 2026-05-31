import csv
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


SPECIAL_TOKENS = {"</s>", "<START>", "<EOP>", "<UNK>", "<PAD>"}
DEFAULT_PROMPTS = list(
    "春江花月夜山水云风雪雨柳松竹梅兰秋夏冬朝暮夕日天人客君酒梦心古长远青白红明寒孤高清碧金玉香鸟鱼舟马"
)


class SequenceDataset(Dataset):
    def __init__(self, sequences):
        self.sequences = [torch.as_tensor(seq, dtype=torch.long) for seq in sequences if len(seq) > 1]

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, index):
        return self.sequences[index]


class PadCollate(object):
    def __init__(self, pad_index):
        self.pad_index = pad_index

    def __call__(self, batch):
        max_len = max(item.size(0) for item in batch)
        padded = batch[0].new_full((len(batch), max_len), self.pad_index)
        for index, item in enumerate(batch):
            padded[index, : item.size(0)] = item
        return padded


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_tang(path="tang.npz"):
    datas = np.load(path, allow_pickle=True)
    data = datas["data"]
    ix2word = datas["ix2word"].item()
    word2ix = datas["word2ix"].item()
    return data, ix2word, word2ix


def strip_left_padding(row, pad_index):
    row = np.asarray(row, dtype=np.int64)
    non_pad = np.flatnonzero(row != pad_index)
    if len(non_pad) == 0:
        return row[-1:]
    return row[non_pad[0] :]


def clean_sequences(data, pad_index):
    return [strip_left_padding(row, pad_index) for row in data]


def split_indices(size, seed=2026, train_ratio=0.8, valid_ratio=0.1):
    rng = np.random.default_rng(seed)
    indices = np.arange(size)
    rng.shuffle(indices)
    train_end = int(size * train_ratio)
    valid_end = train_end + int(size * valid_ratio)
    return {
        "train": indices[:train_end],
        "valid": indices[train_end:valid_end],
        "test": indices[valid_end:],
    }


def text_from_ids(ids, ix2word):
    tokens = [ix2word[int(index)] for index in ids]
    return "".join(token for token in tokens if token not in SPECIAL_TOKENS)


def write_csv(rows, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_generation_log(poems, path, title):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        file.write(f"===== {title} =====\n")
        for poem in poems:
            file.write(poem.strip() + "\n\n")


def available_prompts(word2ix, limit=50):
    prompts = []
    seen = set()
    for prompt in DEFAULT_PROMPTS:
        if prompt in word2ix and prompt not in seen:
            prompts.append(prompt)
            seen.add(prompt)
        if len(prompts) >= limit:
            break
    return prompts
