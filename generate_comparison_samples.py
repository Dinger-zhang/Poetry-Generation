import argparse
import csv
import re
from pathlib import Path

import torch

from config import Config
from experiment_utils import available_prompts, load_tang, set_seed, write_generation_log
from generate import generate
from model import LSTMPoetryModel, TransformerPoetryModel


def clean_gpt2_text(text):
    return re.sub(r"[^\u4e00-\u9fa5，。！？；：、]", "", text).strip()


def run_device():
    if Config.use_gpu and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_char_model(kind, checkpoint, vocab_size, pad_index, device):
    if kind == "lstm":
        model = LSTMPoetryModel(vocab_size, Config.embedding_dim, Config.hidden_dim)
    elif kind == "transformer_raw":
        model = TransformerPoetryModel(vocab_size, Config.embedding_dim)
    elif kind == "transformer_fixed":
        model = TransformerPoetryModel(vocab_size, Config.embedding_dim, padding_idx=pad_index)
    else:
        raise ValueError(f"Unknown char model kind: {kind}")
    model.load_state_dict(torch.load(checkpoint, map_location="cpu"))
    model.to(device)
    model.eval()
    return model


def generate_char_poems(model, prompts, ix2word, word2ix, max_len, sample, temperature, top_k, top_p, penalty):
    old_values = {
        "max_gen_len": Config.max_gen_len,
        "gen_sample": Config.gen_sample,
        "gen_temperature": Config.gen_temperature,
        "gen_top_k": Config.gen_top_k,
        "gen_top_p": Config.gen_top_p,
        "repetition_penalty": Config.repetition_penalty,
    }
    Config.max_gen_len = max_len
    Config.gen_sample = sample
    Config.gen_temperature = temperature
    Config.gen_top_k = top_k
    Config.gen_top_p = top_p
    Config.repetition_penalty = penalty
    try:
        return ["".join(generate(model, prompt, ix2word, word2ix)) for prompt in prompts]
    finally:
        for key, value in old_values.items():
            setattr(Config, key, value)


def load_gpt2(model_dir, device):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(model_dir).to(device)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.eval()
    return model, tokenizer


def generate_gpt2_poems(model, tokenizer, prompts, device, max_len):
    poems = []
    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_len,
                do_sample=True,
                temperature=0.7,
                top_k=50,
                top_p=0.9,
                repetition_penalty=1.2,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        poems.append(clean_gpt2_text(text))
    return poems


def write_sample_csv(rows, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["method", "prompt", "poem"])
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description="Generate comparable poetry samples for all methods.")
    parser.add_argument("--output-dir", default="experiments/generated")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--prompt-limit", type=int, default=50)
    parser.add_argument("--max-new-chars", type=int, default=80)
    parser.add_argument("--lstm", default="checkpoints1/tang_49.pth")
    parser.add_argument("--transformer-raw", default="checkpoints/tang_transformer_run_50.pth")
    parser.add_argument("--transformer-fixed", default="checkpoints/tang_transformer_fixed_50.pth")
    parser.add_argument("--gpt2", default="checkpoints/gpt2_poetry")
    args = parser.parse_args()

    set_seed(args.seed)
    device = run_device()
    data, ix2word, word2ix = load_tang(Config.pickle_path)
    pad_index = word2ix["</s>"]
    prompts = available_prompts(word2ix, args.prompt_limit)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "prompts.txt").write_text("\n".join(prompts) + "\n", encoding="utf-8")

    sample_rows = []

    method_specs = [
        ("LSTM", "lstm", args.lstm),
        ("TransformerRaw", "transformer_raw", args.transformer_raw),
        ("TransformerFixed", "transformer_fixed", args.transformer_fixed),
    ]
    for method_name, kind, checkpoint in method_specs:
        checkpoint_path = Path(checkpoint)
        if not checkpoint_path.exists():
            continue
        model = load_char_model(kind, checkpoint_path, len(word2ix), pad_index, device)
        poems = generate_char_poems(
            model,
            prompts,
            ix2word,
            word2ix,
            args.max_new_chars,
            sample=True,
            temperature=0.9,
            top_k=8,
            top_p=0.9,
            penalty=1.15,
        )
        write_generation_log(poems, output_dir / f"{method_name}.txt", method_name)
        sample_rows.extend(
            {"method": method_name, "prompt": prompt, "poem": poem}
            for prompt, poem in zip(prompts, poems)
        )
        print(f"generated {method_name}: {len(poems)} poems")
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    fixed_checkpoint = Path(args.transformer_fixed)
    if fixed_checkpoint.exists():
        model = load_char_model("transformer_fixed", fixed_checkpoint, len(word2ix), pad_index, device)
        ablations = [
            ("TransformerFixedGreedy", False, 1.0, 0, 1.0, 1.0),
            ("TransformerFixedSampling", True, 1.0, 8, 0.9, 1.0),
            ("TransformerFixedFinal", True, 0.9, 8, 0.9, 1.15),
        ]
        for name, sample, temperature, top_k, top_p, penalty in ablations:
            poems = generate_char_poems(
                model,
                prompts,
                ix2word,
                word2ix,
                args.max_new_chars,
                sample=sample,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                penalty=penalty,
            )
            write_generation_log(poems, output_dir / f"{name}.txt", name)
            print(f"generated {name}: {len(poems)} poems")
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    gpt2_path = Path(args.gpt2)
    if gpt2_path.exists():
        model, tokenizer = load_gpt2(gpt2_path, device)
        poems = generate_gpt2_poems(model, tokenizer, prompts, device, args.max_new_chars)
        write_generation_log(poems, output_dir / "GPT2.txt", "GPT2")
        sample_rows.extend(
            {"method": "GPT2", "prompt": prompt, "poem": poem}
            for prompt, poem in zip(prompts, poems)
        )
        print(f"generated GPT2: {len(poems)} poems")

    write_sample_csv(sample_rows, output_dir / "samples.csv")
    print(f"prompts: {len(prompts)}")
    print(f"output_dir: {output_dir}")


if __name__ == "__main__":
    main()
