import argparse
import csv
import math
import re
from collections import Counter
from pathlib import Path

import numpy as np


SPECIAL_TOKENS = ("</s>", "<START>", "<EOP>")
CHINESE_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]")
LOSS_RE = re.compile(r"^(Epoch\s+\d+\s+)?训练损失为|^=====|^loss\b", re.IGNORECASE)
PUNCTUATION = set("，。！？；、,.!?;:")
END_PUNCTUATION = set("。！？.!?")


def sigmoid_score(value, center, scale):
    if scale <= 0:
        return 1.0 if value == center else 0.0
    return math.exp(-abs(value - center) / scale)


def clamp(value, low=0.0, high=1.0):
    return max(low, min(high, value))


def parse_named_path(value):
    if "=" in value:
        name, path = value.split("=", 1)
        return name.strip(), Path(path.strip())
    path = Path(value)
    return path.stem, path


def is_log_marker(line):
    return not line or LOSS_RE.search(line) is not None


def read_generated_poems(path, last_group_only=True):
    groups = []
    current = []

    for raw_line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if is_log_marker(line):
            if current:
                groups.append(current)
                current = []
            continue
        current.append(line)

    if current:
        groups.append(current)

    if not groups:
        return []
    return groups[-1] if last_group_only else [poem for group in groups for poem in group]


def remove_special_tokens(text):
    for token in SPECIAL_TOKENS:
        text = text.replace(token, "")
    return text.strip()


def chinese_chars(text):
    return CHINESE_RE.findall(text)


def split_lines(text):
    parts = re.split(r"[，。！？；,.!?;]+", text)
    return [chinese_chars(part) for part in parts if chinese_chars(part)]


def ngrams(items, n):
    if len(items) < n:
        return []
    return [tuple(items[i : i + n]) for i in range(len(items) - n + 1)]


def distinct_n(chars, n):
    grams = ngrams(chars, n)
    if not grams:
        return 1.0
    return len(set(grams)) / len(grams)


def max_run_length(chars):
    if not chars:
        return 0
    best = 1
    current = 1
    for left, right in zip(chars, chars[1:]):
        if left == right:
            current += 1
            best = max(best, current)
        else:
            current = 1
    return best


def js_similarity(generated_counts, corpus_counts):
    generated_total = sum(generated_counts.values())
    corpus_total = sum(corpus_counts.values())
    if generated_total == 0 or corpus_total == 0:
        return 0.0, 1.0

    keys = set(generated_counts) | set(corpus_counts)
    js_divergence = 0.0
    for key in keys:
        p = generated_counts.get(key, 0) / generated_total
        q = corpus_counts.get(key, 0) / corpus_total
        m = 0.5 * (p + q)
        if p > 0:
            js_divergence += 0.5 * p * math.log(p / m)
        if q > 0:
            js_divergence += 0.5 * q * math.log(q / m)

    normalized = js_divergence / math.log(2)
    return clamp(1.0 - normalized), normalized


def load_corpus(path):
    datas = np.load(path, allow_pickle=True)
    data = datas["data"]
    ix2word = datas["ix2word"].item()

    corpus_counts = Counter()
    corpus_8grams = set()
    lengths = []

    for row in data:
        tokens = [ix2word[int(index)] for index in row]
        text = "".join(token for token in tokens if token not in SPECIAL_TOKENS)
        chars = chinese_chars(text)
        if not chars:
            continue
        corpus_counts.update(chars)
        corpus_8grams.update(ngrams(chars, 8))
        lengths.append(len(chars))

    if lengths:
        median_len = float(np.median(lengths))
        q1 = float(np.percentile(lengths, 25))
        q3 = float(np.percentile(lengths, 75))
        iqr = max(1.0, q3 - q1)
    else:
        median_len = 40.0
        iqr = 20.0

    return {
        "counts": corpus_counts,
        "vocab": set(corpus_counts),
        "8grams": corpus_8grams,
        "median_len": median_len,
        "length_scale": max(10.0, 1.5 * iqr),
    }


def poem_metrics(text, corpus):
    raw_text = text.strip()
    special_count = sum(raw_text.count(token) for token in SPECIAL_TOKENS)
    clean_text = remove_special_tokens(raw_text)
    chars = chinese_chars(clean_text)
    char_count = len(chars)
    char_counter = Counter(chars)

    special_ratio = special_count / max(1, char_count + special_count)
    oov_count = sum(1 for char in chars if char not in corpus["vocab"])
    oov_ratio = oov_count / max(1, char_count)
    clean_score = clamp(1.0 - 4.0 * special_ratio - 2.0 * oov_ratio)

    line_chars = split_lines(clean_text)
    line_lengths = [len(item) for item in line_chars]
    if line_lengths:
        fixed_line_ratio = sum(1 for length in line_lengths if length in (5, 7)) / len(line_lengths)
        line_length_score = sum(
            clamp(1.0 - min(abs(length - 5), abs(length - 7)) / 7.0)
            for length in line_lengths
        ) / len(line_lengths)
        pairs = list(zip(line_lengths[0::2], line_lengths[1::2]))
        pair_match_ratio = (
            sum(1 for left, right in pairs if left == right and left in (5, 7)) / len(pairs)
            if pairs
            else fixed_line_ratio
        )
    else:
        fixed_line_ratio = 0.0
        line_length_score = 0.0
        pair_match_ratio = 0.0

    punctuation_count = sum(1 for char in clean_text if char in PUNCTUATION)
    punctuation_density = punctuation_count / max(1, len(clean_text))
    punctuation_score = clamp(1.0 - abs(punctuation_density - 0.14) / 0.14)
    ending_score = 1.0 if clean_text and clean_text[-1] in END_PUNCTUATION else 0.0
    form_score = (
        0.35 * fixed_line_ratio
        + 0.25 * line_length_score
        + 0.20 * pair_match_ratio
        + 0.10 * punctuation_score
        + 0.10 * ending_score
    )

    adjacent_repeat = (
        sum(1 for left, right in zip(chars, chars[1:]) if left == right) / max(1, char_count - 1)
    )
    repeat_2gram = 1.0 - distinct_n(chars, 2)
    repeat_3gram = 1.0 - distinct_n(chars, 3)
    run_penalty = clamp((max_run_length(chars) - 2) / 6.0)
    repetition_penalty = (
        0.30 * adjacent_repeat
        + 0.25 * repeat_2gram
        + 0.25 * repeat_3gram
        + 0.20 * run_penalty
    )
    fluency_score = clamp(1.0 - 2.0 * repetition_penalty)

    length_score = sigmoid_score(char_count, corpus["median_len"], corpus["length_scale"])

    grams8 = ngrams(chars, 8)
    if grams8:
        corpus_overlap_8gram = sum(1 for gram in grams8 if gram in corpus["8grams"]) / len(grams8)
    else:
        corpus_overlap_8gram = 0.0
    novelty_score = clamp(1.0 - corpus_overlap_8gram)

    return {
        "text": clean_text,
        "char_count": char_count,
        "char_counter": char_counter,
        "clean_score": clean_score,
        "form_score": form_score,
        "fluency_score": fluency_score,
        "length_score": length_score,
        "novelty_score": novelty_score,
        "special_ratio": special_ratio,
        "oov_ratio": oov_ratio,
        "fixed_line_ratio": fixed_line_ratio,
        "punctuation_density": punctuation_density,
        "distinct_1": distinct_n(chars, 1),
        "distinct_2": distinct_n(chars, 2),
        "adjacent_repeat": adjacent_repeat,
        "corpus_overlap_8gram": corpus_overlap_8gram,
    }


def average(values):
    values = list(values)
    if not values:
        return 0.0
    return sum(values) / len(values)


def evaluate_method(name, poems, corpus):
    per_poem = [poem_metrics(poem, corpus) for poem in poems]
    generated_counts = Counter()
    for item in per_poem:
        generated_counts.update(item["char_counter"])

    style_score, js_divergence = js_similarity(generated_counts, corpus["counts"])

    clean_score = average(item["clean_score"] for item in per_poem)
    form_score = average(item["form_score"] for item in per_poem)
    fluency_score = average(item["fluency_score"] for item in per_poem)
    length_score = average(item["length_score"] for item in per_poem)
    novelty_score = average(item["novelty_score"] for item in per_poem)

    overall_score = 100.0 * (
        0.15 * clean_score
        + 0.25 * form_score
        + 0.25 * fluency_score
        + 0.15 * style_score
        + 0.10 * length_score
        + 0.10 * novelty_score
    )

    return {
        "method": name,
        "poems": len(poems),
        "overall": overall_score,
        "clean": 100.0 * clean_score,
        "form": 100.0 * form_score,
        "fluency": 100.0 * fluency_score,
        "style": 100.0 * style_score,
        "length": 100.0 * length_score,
        "novelty": 100.0 * novelty_score,
        "avg_chars": average(item["char_count"] for item in per_poem),
        "special_ratio": average(item["special_ratio"] for item in per_poem),
        "oov_ratio": average(item["oov_ratio"] for item in per_poem),
        "fixed_line_ratio": average(item["fixed_line_ratio"] for item in per_poem),
        "distinct_1": average(item["distinct_1"] for item in per_poem),
        "distinct_2": average(item["distinct_2"] for item in per_poem),
        "adjacent_repeat": average(item["adjacent_repeat"] for item in per_poem),
        "js_divergence": js_divergence,
        "corpus_overlap_8gram": average(item["corpus_overlap_8gram"] for item in per_poem),
    }


def format_table(rows, columns):
    widths = {}
    for column in columns:
        widths[column] = max(len(column), *(len(format_value(row[column])) for row in rows))

    header = "  ".join(column.ljust(widths[column]) for column in columns)
    separator = "  ".join("-" * widths[column] for column in columns)
    lines = [header, separator]
    for row in rows:
        lines.append("  ".join(format_value(row[column]).ljust(widths[column]) for column in columns))
    return "\n".join(lines)


def format_value(value):
    if isinstance(value, float):
        return f"{value:.4f}" if abs(value) < 1 else f"{value:.2f}"
    return str(value)


def write_csv(rows, path):
    if not rows:
        return
    columns = list(rows[0].keys())
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate generated poetry files with one model-independent metric suite."
    )
    parser.add_argument(
        "files",
        nargs="+",
        help="Generated result files. Use name=path to set the method name.",
    )
    parser.add_argument("--corpus", default="tang.npz", help="Reference corpus npz file.")
    parser.add_argument(
        "--all",
        action="store_true",
        help="Evaluate all generated poems in each log instead of only the last group.",
    )
    parser.add_argument("--output", help="Optional CSV output path.")
    args = parser.parse_args()

    corpus = load_corpus(args.corpus)
    rows = []
    for file_arg in args.files:
        name, path = parse_named_path(file_arg)
        poems = read_generated_poems(path, last_group_only=not args.all)
        if not poems:
            raise ValueError(f"No generated poems found in {path}")
        rows.append(evaluate_method(name, poems, corpus))

    rows.sort(key=lambda row: row["overall"], reverse=True)
    summary_columns = [
        "method",
        "poems",
        "overall",
        "clean",
        "form",
        "fluency",
        "style",
        "length",
        "novelty",
        "avg_chars",
        "special_ratio",
        "fixed_line_ratio",
        "adjacent_repeat",
    ]
    print(format_table(rows, summary_columns))

    if args.output:
        output_path = Path(args.output)
        write_csv(rows, output_path)
        print(f"\nSaved CSV report to {output_path}")


if __name__ == "__main__":
    main()
