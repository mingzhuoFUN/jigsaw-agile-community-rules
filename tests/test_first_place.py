from pathlib import Path

import pandas as pd

from src.first_place.data import build_training_frame
from src.first_place.ensemble import blend


def test_training_frame_repeats_test_examples(tmp_path: Path) -> None:
    pd.DataFrame(
        [{"body": "base", "rule": "rule", "rule_violation": 1}]
    ).to_csv(tmp_path / "train.csv", index=False)
    pd.DataFrame([{
        "positive_example_1": "p1", "positive_example_2": "p2",
        "negative_example_1": "n1", "negative_example_2": "n2", "rule": "rule",
    }]).to_csv(tmp_path / "test.csv", index=False)

    frame = build_training_frame(tmp_path, example_repeats=3)
    assert len(frame) == 13
    assert (frame["source"] == "test_examples").sum() == 12


def test_blend_normalizes_notebook_weights(tmp_path: Path) -> None:
    pd.DataFrame({"row_id": [1], "rule_violation": [0.0]}).to_csv(
        tmp_path / "submission1.csv", index=False
    )
    pd.DataFrame({"row_id": [1], "rule_violation": [1.0]}).to_csv(
        tmp_path / "submission2.csv", index=False
    )
    result = blend(
        tmp_path,
        tmp_path / "out.csv",
        {"submission1.csv": 0.2, "submission2.csv": 0.2},
    )
    assert result.loc[0, "rule_violation"] == 0.5

