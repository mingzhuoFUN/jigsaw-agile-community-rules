from __future__ import annotations

from pathlib import Path

import pandas as pd


PROMPT = "Reddit moderation: Does the comment violate the rule? Answer 'Yes' or 'No' only."
TRAIN_COLUMNS = {"body", "rule", "rule_violation"}
TEST_COLUMNS = {
    "row_id",
    "body",
    "rule",
    "positive_example_1",
    "positive_example_2",
    "negative_example_1",
    "negative_example_2",
}
TEXT_COLUMNS = ("body", "rule")


def build_prompt(body: str, rule: str) -> str:
    return f"{PROMPT}\n\nComment: {body}\n\nRule: {rule}\n---\nAnswer:"


def _require_columns(frame: pd.DataFrame, required: set[str], filename: str) -> None:
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{filename} is missing required columns: {missing}")


def _clean_text(series: pd.Series) -> pd.Series:
    """Apply conservative cleaning without rewriting the meaning of a comment or rule."""
    return (
        series.astype("string")
        .str.replace("\r\n", "\n", regex=False)
        .str.replace("\r", "\n", regex=False)
        .str.replace("\x00", "", regex=False)
        .str.strip()
    )


def _clean_labeled_rows(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    for column in TEXT_COLUMNS:
        result[column] = _clean_text(result[column])
    result["rule_violation"] = pd.to_numeric(result["rule_violation"], errors="coerce")
    result = result.dropna(subset=[*TEXT_COLUMNS, "rule_violation"])
    result = result[
        result["body"].str.len().gt(0)
        & result["rule"].str.len().gt(0)
        & result["rule_violation"].isin([0, 1])
    ]
    result["rule_violation"] = result["rule_violation"].astype("int8")
    return result


def build_training_frame(
    data_dir: Path,
    example_repeats: int = 2,
    seed: int = 1001,
) -> pd.DataFrame:
    """Build the winner-style labeled frame from train rows and test demonstrations.

    ``example_repeats=2`` matches the executable reference notebook: each unique
    positive/negative test demonstration appears once in the initial frame and is
    appended once more. A different value is an explicit experiment, not a
    faithful-reproduction setting.
    """
    if example_repeats < 1:
        raise ValueError("example_repeats must be at least 1")

    train = pd.read_csv(data_dir / "train.csv")
    test = pd.read_csv(data_dir / "test.csv")
    _require_columns(train, TRAIN_COLUMNS, "train.csv")
    _require_columns(test, TEST_COLUMNS, "test.csv")

    base = _clean_labeled_rows(
        train[["body", "rule", "rule_violation"]].assign(source="train")
    )

    examples = []
    for polarity, label in (("positive", 1), ("negative", 0)):
        for number in (1, 2):
            column = f"{polarity}_example_{number}"
            frame = test[[column, "rule"]].rename(columns={column: "body"})
            examples.append(frame.assign(rule_violation=label, source="test_examples"))

    example_frame = _clean_labeled_rows(pd.concat(examples, ignore_index=True))
    example_frame = example_frame.drop_duplicates(
        ["body", "rule", "rule_violation", "source"],
        ignore_index=True,
    )
    base = base.drop_duplicates(
        ["body", "rule", "rule_violation", "source"],
        ignore_index=True,
    )
    result = pd.concat([base, example_frame], ignore_index=True)

    if example_repeats > 1:
        result = pd.concat(
            [result, *([example_frame] * (example_repeats - 1))],
            ignore_index=True,
        )
    return result.sample(frac=1, random_state=seed).reset_index(drop=True)


def build_classification_text(frame: pd.DataFrame) -> pd.Series:
    return frame.apply(lambda row: build_prompt(str(row["body"]), str(row["rule"])), axis=1)
