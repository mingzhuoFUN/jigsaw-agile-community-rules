from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def validate_submission(path: Path, data_dir: Path) -> pd.DataFrame:
    """Validate the Kaggle submission contract and return it in sample order."""
    submission = pd.read_csv(path)
    sample = pd.read_csv(data_dir / "sample_submission.csv")
    if list(submission.columns) != ["row_id", "rule_violation"]:
        raise ValueError(f"Unexpected submission columns: {submission.columns.tolist()}")
    if len(submission) != len(sample):
        raise ValueError(f"Expected {len(sample)} predictions, got {len(submission)}")
    if set(submission["row_id"]) != set(sample["row_id"]):
        raise ValueError("Submission row_id values do not match sample_submission.csv")
    if submission["row_id"].duplicated().any():
        raise ValueError("Duplicate row_id values in submission")
    predictions = pd.to_numeric(submission["rule_violation"], errors="coerce")
    if not np.isfinite(predictions).all():
        raise ValueError("Submission contains missing or non-finite predictions")
    if not predictions.between(0, 1).all():
        raise ValueError("Predictions must be in [0, 1]")

    ordered = sample[["row_id"]].merge(
        submission.assign(rule_violation=predictions),
        on="row_id",
        validate="one_to_one",
    )
    ordered.to_csv(path, index=False)
    return ordered
