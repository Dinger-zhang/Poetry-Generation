# Poetry Generation Experiment Summary

## Data Split

- Seed: 2026
- Train / valid / test: 8:1:1
- Train: 46,064 poems
- Valid: 5,758 poems
- Test: 5,758 poems
- Prompt set: 49 common Tang-poetry imagery characters

## Unified Validation Loss

| Method | Protocol | Loss | Perplexity |
| --- | --- | ---: | ---: |
| LSTM | clean valid, character-level, ignore padding | 4.9427 | 140.14 |
| TransformerRaw | clean valid, character-level, ignore padding | 4.0708 | 58.61 |
| TransformerFixed | clean valid, character-level, ignore padding | 2.7099 | 15.03 |
| GPT2 | clean valid, BPE-token-level | 3.1712 | 23.84 |
| LSTM | raw left-padded valid, diagnostic | 1.0469 | 2.85 |
| TransformerRaw | raw left-padded valid, diagnostic | 3.5451 | 34.64 |

## Automatic Text Quality, 49 Prompts

| Method | Overall | Clean | Form | Fluency | Style | Length | Novelty |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| TransformerFixed | 91.11 | 100.00 | 96.65 | 99.71 | 67.64 | 68.74 | 100.00 |
| LSTM | 88.07 | 100.00 | 82.97 | 99.40 | 82.42 | 51.33 | 99.76 |
| GPT2 | 84.13 | 78.60 | 82.09 | 99.96 | 59.39 | 79.17 | 100.00 |

## Transformer Padding Ablation

| Method | Overall | Form | Length | Avg chars |
| --- | ---: | ---: | ---: | ---: |
| TransformerRaw | 85.79 | 80.11 | 52.47 | 67.10 |
| TransformerFixed | 91.11 | 96.65 | 68.74 | 23.80 |

## Decoding Ablation

| Decoding | Overall | Form | Fluency | Length |
| --- | ---: | ---: | ---: | ---: |
| Greedy | 88.62 | 94.38 | 96.48 | 65.26 |
| Sampling | 91.36 | 96.49 | 99.39 | 70.89 |
| Final | 91.92 | 98.03 | 99.85 | 70.96 |

## Key Conclusion

The low LSTM training loss is not sufficient evidence that LSTM is better. On the clean character-level validation split, TransformerFixed has the lowest validation loss and the best automatic text-quality score. Padding handling and decoding strategy materially change Transformer output quality.
