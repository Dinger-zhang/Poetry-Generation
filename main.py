# main.py

import os
import torch
import numpy as np
from transformers import (
    GPT2LMHeadModel,
    GPT2Tokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling
)
from torch.utils.data import Dataset
from config import Config
import warnings
warnings.filterwarnings("ignore")

# ---------- 数据集类 ----------
class PoetryDataset(Dataset):
    def __init__(self, input_ids_list, attention_mask_list):
        self.input_ids = input_ids_list
        self.attention_mask = attention_mask_list

    def __getitem__(self, idx):
        return {
            'input_ids': self.input_ids[idx],
            'attention_mask': self.attention_mask[idx]
        }

    def __len__(self):
        return len(self.input_ids)

def load_and_prepare_data(config):
    """加载 tang.npz，将整数序列转成文本，再用 GPT‑2 tokenizer 编码"""
    print("加载 tang.npz ...")
    datas = np.load(config.pickle_path, allow_pickle=True)
    data = datas['data']            # (N, maxlen) 整数矩阵
    ix2word = datas['ix2word'].item()
    word2ix = datas['word2ix'].item()

    print("将整数序列转换为文本 ...")
    texts = []
    special_tokens = {'<START>', '<EOP>', '<UNK>', '<PAD>'}
    for seq in data:
        chars = []
        for idx in seq:
            if idx == 0:            # 假定 0 为填充符
                continue
            word = ix2word[idx]
            if word in special_tokens:
                continue
            chars.append(word)
        if chars:                   # 忽略空序列
            texts.append(''.join(chars))
    print(f"共加载 {len(texts)} 首有效诗歌")
    import random
    texts = random.sample(texts, 10000)
    print(f"采样后使用 {len(texts)} 首诗歌进行微调")
    # 加载 GPT‑2 tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained(config.gpt2_model_name)
    # GPT‑2 原始没有 pad_token，设置 eos_token 作为 pad_token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 编码所有文本（返回字典，键为 'input_ids', 'attention_mask'）
    encodings = tokenizer(
        texts,
        truncation=True,
        max_length=config.gpt2_max_length,
        padding=False,           # 动态 padding 由 DataCollator 处理
        return_tensors=None      # 返回 Python 列表
    )
    # 转换为 PyTorch tensor 列表（每个元素是一个样本的 tensor）
    input_ids = [torch.tensor(seq, dtype=torch.long) for seq in encodings['input_ids']]
    attention_masks = [torch.tensor(mask, dtype=torch.long) for mask in encodings['attention_mask']]

    dataset = PoetryDataset(input_ids, attention_masks)
    return dataset, tokenizer

def train():
    # 禁用 wandb 等日志工具

    os.environ["WANDB_DISABLED"] = "true"

    # 设备设置
    if Config.use_gpu and torch.cuda.is_available():
        Config.device = torch.device("cuda")
    else:
        Config.device = torch.device("cpu")
    print(f"使用设备: {Config.device}")
    # 创建输出目录
    os.makedirs(Config.gpt2_output_dir, exist_ok=True)

    # 准备数据集和 tokenizer
    dataset, tokenizer = load_and_prepare_data(Config)

    # 加载 GPT‑2 模型
    model = GPT2LMHeadModel.from_pretrained(Config.gpt2_model_name)
    model.resize_token_embeddings(len(tokenizer))   # 确保 embedding 大小与 tokenizer 一致
    model.to(Config.device)

    # 数据整理器（动态 padding 和语言建模）
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,                     # 因果语言模型不使用 MLM
        pad_to_multiple_of=8
    )

    # 训练参数（兼容新旧版本 transformers）
    training_args = TrainingArguments(
        output_dir=Config.gpt2_output_dir,
        overwrite_output_dir=Config.gpt2_overwrite_output_dir,
        num_train_epochs=Config.epoch,
        per_device_train_batch_size=Config.batch_size,
        gradient_accumulation_steps=2,
        learning_rate=Config.gpt2_learning_rate,
        weight_decay=Config.gpt2_weight_decay,
        warmup_steps=Config.gpt2_warmup_steps,
        logging_steps=Config.gpt2_logging_steps,
        save_steps=Config.gpt2_save_steps,
        fp16=Config.gpt2_fp16,
        save_total_limit=3,
        dataloader_num_workers=4,
        logging_strategy="epoch"
        # 移除 evaluation_strategy（没有验证集）
        # 移除 report_to（避免 wandb 相关报错）
    )

    # 创建 Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=dataset,
        tokenizer=tokenizer,
    )

    # 开始训练
    trainer.train()

    # 保存最终模型
    trainer.save_model(Config.gpt2_output_dir)
    tokenizer.save_pretrained(Config.gpt2_output_dir)
    print(f"微调完成，模型已保存至 {Config.gpt2_output_dir}")

if __name__ == '__main__':
    train()