import argparse
import csv
import random
from pathlib import Path


def read_samples(path):
    rows = []
    with Path(path).open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        for row in reader:
            if row.get("poem", "").strip():
                rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser(description="Create blinded manual-evaluation sheets.")
    parser.add_argument("--samples", default="experiments/generated/samples.csv")
    parser.add_argument("--output", default="experiments/manual_eval_template.csv")
    parser.add_argument("--key", default="experiments/manual_eval_key.csv")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--prompts", type=int, default=10)
    parser.add_argument("--methods", default="LSTM,TransformerFixed,GPT2")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    rows = read_samples(args.samples)
    methods = {item.strip() for item in args.methods.split(",") if item.strip()}
    rows = [row for row in rows if row.get("method") in methods]
    prompts = sorted({row["prompt"] for row in rows})
    selected_prompts = rng.sample(prompts, min(args.prompts, len(prompts)))
    selected_rows = [row for row in rows if row["prompt"] in selected_prompts]
    rng.shuffle(selected_rows)

    template_rows = []
    key_rows = []
    for index, row in enumerate(selected_rows, start=1):
        sample_id = f"S{index:03d}"
        template_rows.append(
            {
                "sample_id": sample_id,
                "prompt": row["prompt"],
                "poem": row["poem"],
                "format_score_1_5": "",
                "fluency_score_1_5": "",
                "style_score_1_5": "",
                "imagery_score_1_5": "",
                "overall_score_1_5": "",
                "comment": "",
            }
        )
        key_rows.append(
            {
                "sample_id": sample_id,
                "prompt": row["prompt"],
                "method": row["method"],
            }
        )

    for path, output_rows in [(Path(args.output), template_rows), (Path(args.key), key_rows)]:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8-sig", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=list(output_rows[0].keys()))
            writer.writeheader()
            writer.writerows(output_rows)

    print(f"manual samples: {len(template_rows)}")
    print(f"template: {args.output}")
    print(f"key: {args.key}")


if __name__ == "__main__":
    main()
