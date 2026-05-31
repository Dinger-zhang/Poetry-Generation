import argparse
from pathlib import Path

from experiment_utils import clean_sequences, load_tang, split_indices, text_from_ids, write_csv


def main():
    parser = argparse.ArgumentParser(description="Export clean text splits for GPT-2 fine-tuning.")
    parser.add_argument("--data", default="tang.npz")
    parser.add_argument("--output-dir", default="data/gpt2_clean")
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    data, ix2word, word2ix = load_tang(args.data)
    pad_index = word2ix["</s>"]
    sequences = clean_sequences(data, pad_index)
    splits = split_indices(len(sequences), seed=args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for split_name, indices in splits.items():
        texts = [text_from_ids(sequences[index], ix2word) for index in indices]
        texts = [text for text in texts if text.strip()]
        output_path = output_dir / f"{split_name}.txt"
        output_path.write_text("\n".join(texts) + "\n", encoding="utf-8")
        lengths = [len(text) for text in texts]
        rows.append(
            {
                "split": split_name,
                "samples": len(texts),
                "min_chars": min(lengths) if lengths else 0,
                "max_chars": max(lengths) if lengths else 0,
                "avg_chars": sum(lengths) / max(1, len(lengths)),
                "path": str(output_path),
            }
        )

    write_csv(rows, output_dir / "stats.csv")
    for row in rows:
        print(
            f"{row['split']:<5} samples={row['samples']} "
            f"avg_chars={row['avg_chars']:.2f} path={row['path']}"
        )


if __name__ == "__main__":
    main()
