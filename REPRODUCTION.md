# First-place reproduction contract

## Reference

- Source: `1st-place-code (1).ipynb`
- Notebook format: 4.5, 31 cells
- Frozen script payloads: `reference/notebook_cells/`
- Environment-adapted runnable copies: `scripts/winner/`
- Competition data: Kaggle `jigsaw-agile-community-rules`

The repository preserves notebook semantics. Changes in `scripts/winner/` are limited to
portable paths, Hugging Face model identifiers, and online model loading.

## Actual active run

The notebook's `train.py` actively runs six predictions:

| Output | Model | Seed |
|---|---|---:|
| submission1 | Qwen3 14B | 1001 |
| submission2 | Qwen2.5 14B | 1004 |
| submission3 | Qwen3 8B | 1009 |
| submission4 | Llama 3.1 8B | 1003 |
| submission5 | Qwen3 4B | 1010 |
| submission7 | Ettin encoder 400M | 3001 |

Phi-4 and Llama 3.2 3B are defined but commented out. They remain available in
`scripts/winner/` and are not silently added to the faithful run.

## Training and inference

- Training rows are combined with labeled positive/negative examples supplied in the test set.
- The executable notebook contains each unique target example twice after de-duplication
  (the nearby source comment incorrectly says three times).
- Generative models use 4-bit LoRA through Unsloth, one epoch, response-only loss, and
  256-token training / 512-token inference limits.
- Ettin uses sequence classification and produces sigmoid probabilities.
- Predictions are aligned by `row_id`.

## Reference ambiguity retained

The active notebook blend is:

```text
0.2*s1 + 0.2*s2 + 0.2*s3 + 0.2*s4 + 0.2*s5 + 0.1*s7
```

These coefficients sum to `1.1`. `src.first_place.ensemble` normalizes them by default
for a valid probability scale; this is documented as a correction, not presented as an
exact reference result.

## Expected result

The supplied notebook contains no leaderboard score or local validation score. A faithful
score comparison can therefore only be made after submitting `submission.csv` to the same
Kaggle competition/evaluation environment. No local AUC is claimed to match the winner.

## Recommended runnable contract

The supported Colab proof uses the Ettin-400M component only. It preserves the reference
data construction, rule-aware pair classification, per-rule ranking, and submission
contract while avoiding the cost and memory requirements of the complete ensemble.
Successful execution proves the GitHub -> Colab -> Hugging Face -> training -> inference
-> Google Drive path; it does not claim the first-place ensemble score.

## Artifacts

Colab writes logs, per-model submissions, and the final blend under Google Drive. Model
checkpoints created by individual scripts should also be moved there before the runtime ends.
