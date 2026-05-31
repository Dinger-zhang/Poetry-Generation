import argparse
import math
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from config import Config
from experiment_utils import (
    PadCollate,
    SequenceDataset,
    clean_sequences,
    load_tang,
    split_indices,
    text_from_ids,
    write_csv,
)
from model import LSTMPoetryModel, TransformerPoetryModel


def device():
    if Config.use_gpu and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_state(model, path):
    state = torch.load(path, map_location="cpu")
    model.load_state_dict(state)
    return model


def evaluate_char_model(model, sequences, pad_index, batch_size, run_device):
    dataset = SequenceDataset(sequences)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=PadCollate(pad_index),
    )
    criterion = nn.CrossEntropyLoss(ignore_index=pad_index, reduction="sum")
    model.to(run_device)
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for batch in loader:
            batch = batch.long().transpose(1, 0).contiguous().to(run_device)
            input_, target = batch[:-1], batch[1:]
            output, _ = model(input_)
            target_flat = target.contiguous().view(-1)
            loss = criterion(output, target_flat)
            total_loss += float(loss.detach().cpu())
            total_tokens += int(target_flat.ne(pad_index).sum().detach().cpu())

    avg_loss = total_loss / max(1, total_tokens)
    return avg_loss, math.exp(min(avg_loss, 20)), total_tokens, len(dataset)


def evaluate_gpt2(model_dir, texts, batch_size, run_device):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(model_dir).to(run_device)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.eval()

    criterion = nn.CrossEntropyLoss(ignore_index=-100, reduction="sum")
    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            batch_texts = texts[start : start + batch_size]
            encoded = tokenizer(
                batch_texts,
                truncation=True,
                max_length=128,
                padding=True,
                return_tensors="pt",
            ).to(run_device)
            outputs = model(**encoded)
            logits = outputs.logits[:, :-1, :].contiguous()
            labels = encoded["input_ids"][:, 1:].contiguous()
            mask = encoded["attention_mask"][:, 1:].contiguous()
            labels = labels.masked_fill(mask.eq(0), -100)
            loss = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))
            total_loss += float(loss.detach().cpu())
            total_tokens += int(labels.ne(-100).sum().detach().cpu())

    avg_loss = total_loss / max(1, total_tokens)
    return avg_loss, math.exp(min(avg_loss, 20)), total_tokens, len(texts)


def main():
    parser = argparse.ArgumentParser(description="Run unified validation-loss experiments.")
    parser.add_argument("--data", default=Config.pickle_path)
    parser.add_argument("--output", default="experiments/validation_losses.csv")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--gpt2-batch-size", type=int, default=8)
    parser.add_argument("--lstm", default="checkpoints1/tang_49.pth")
    parser.add_argument("--transformer-raw", default="checkpoints/tang_transformer_run_50.pth")
    parser.add_argument("--transformer-fixed", default="checkpoints/tang_transformer_fixed_50.pth")
    parser.add_argument("--gpt2", default="checkpoints/gpt2_poetry")
    args = parser.parse_args()

    run_device = device()
    data, ix2word, word2ix = load_tang(args.data)
    pad_index = word2ix["</s>"]
    splits = split_indices(len(data), seed=args.seed)
    valid_raw = [data[index] for index in splits["valid"]]
    valid_clean = clean_sequences(data[splits["valid"]], pad_index)
    valid_texts = [text_from_ids(seq, ix2word) for seq in valid_clean]

    rows = []

    char_jobs = [
        {
            "method": "LSTM",
            "checkpoint": args.lstm,
            "model": LSTMPoetryModel(len(word2ix), Config.embedding_dim, Config.hidden_dim),
            "sequences": valid_clean,
            "protocol": "clean_valid_ignore_pad",
        },
        {
            "method": "TransformerRaw",
            "checkpoint": args.transformer_raw,
            "model": TransformerPoetryModel(len(word2ix), Config.embedding_dim),
            "sequences": valid_clean,
            "protocol": "clean_valid_ignore_pad",
        },
        {
            "method": "TransformerFixed",
            "checkpoint": args.transformer_fixed,
            "model": TransformerPoetryModel(len(word2ix), Config.embedding_dim, padding_idx=pad_index),
            "sequences": valid_clean,
            "protocol": "clean_valid_ignore_pad",
        },
        {
            "method": "LSTM",
            "checkpoint": args.lstm,
            "model": LSTMPoetryModel(len(word2ix), Config.embedding_dim, Config.hidden_dim),
            "sequences": valid_raw,
            "protocol": "raw_left_padded_ignore_pad",
        },
        {
            "method": "TransformerRaw",
            "checkpoint": args.transformer_raw,
            "model": TransformerPoetryModel(len(word2ix), Config.embedding_dim),
            "sequences": valid_raw,
            "protocol": "raw_left_padded_ignore_pad",
        },
    ]

    for job in char_jobs:
        checkpoint = Path(job["checkpoint"])
        if not checkpoint.exists():
            continue
        model = load_state(job["model"], checkpoint)
        loss, ppl, tokens, samples = evaluate_char_model(
            model, job["sequences"], pad_index, args.batch_size, run_device
        )
        rows.append(
            {
                "method": job["method"],
                "protocol": job["protocol"],
                "checkpoint": str(checkpoint),
                "samples": samples,
                "tokens": tokens,
                "loss": loss,
                "perplexity": ppl,
            }
        )

    gpt2_path = Path(args.gpt2)
    if gpt2_path.exists():
        loss, ppl, tokens, samples = evaluate_gpt2(gpt2_path, valid_texts, args.gpt2_batch_size, run_device)
        rows.append(
            {
                "method": "GPT2",
                "protocol": "clean_valid_bpe_token_loss",
                "checkpoint": str(gpt2_path),
                "samples": samples,
                "tokens": tokens,
                "loss": loss,
                "perplexity": ppl,
            }
        )

    write_csv(rows, args.output)
    for row in rows:
        print(
            f"{row['method']:<17} {row['protocol']:<27} "
            f"loss={row['loss']:.4f} ppl={row['perplexity']:.2f} tokens={row['tokens']}"
        )


if __name__ == "__main__":
    main()
