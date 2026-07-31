from __future__ import annotations

from pathlib import Path

import pandas as pd


PROMPT = "Reddit moderation: Does the comment violate the rule? Answer 'Yes' or 'No' only."


def build_prompt(body: str, rule: str) -> str:
    return f"{PROMPT}\n\nComment: {body}\n\nRule: {rule}\n---\nAnswer:"


def build_training_frame(data_dir: Path, example_repeats: int = 3, seed: int = 1001) -> pd.DataFrame:
    """Reproduce the winner's training set, including labeled test examples."""
    if example_repeats < 1:
        raise ValueError("example_repeats must be at least 1")

    train = pd.read_csv(data_dir / "train.csv")
    test = pd.read_csv(data_dir / "test.csv")
    frames = [
        train[["body", "rule", "rule_violation"]].assign(source="train")
    ]

    examples = []
    for polarity, label in (("positive", 1), ("negative", 0)):
        for number in (1, 2):
            column = f"{polarity}_example_{number}"
            frame = test[[column, "rule"]].rename(columns={column: "body"})
            examples.append(frame.assign(rule_violation=label, source="test_examples"))

    example_frame = pd.concat(examples, ignore_index=True).dropna(subset=["body", "rule"])
    frames.extend([example_frame] * example_repeats)
    result = pd.concat(frames, ignore_index=True)
    result = result.drop_duplicates(["body", "rule", "rule_violation", "source"])

    # Repeat only the target-domain examples after de-duplication, matching the notebook.
    if example_repeats > 1:
        result = pd.concat(
            [result, *([result[result["source"] == "test_examples"]] * (example_repeats - 1))],
            ignore_index=True,
        )
    return result.sample(frac=1, random_state=seed).reset_index(drop=True)


def build_classification_text(frame: pd.DataFrame) -> pd.Series:
    return frame.apply(lambda row: build_prompt(str(row["body"]), str(row["rule"])), axis=1)

