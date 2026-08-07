from pathlib import Path

import pandas as pd

from first_place.data import build_training_frame
from first_place.ensemble import blend
from first_place.validation import validate_submission


def test_training_frame_matches_reference_example_repeats(tmp_path: Path) -> None:
    pd.DataFrame(
        [{"body": "base", "rule": "rule", "rule_violation": 1}]
    ).to_csv(tmp_path / "train.csv", index=False)
    pd.DataFrame([{
        "row_id": 1, "body": "target",
        "positive_example_1": "p1", "positive_example_2": "p2",
        "negative_example_1": "n1", "negative_example_2": "n2", "rule": "rule",
    }]).to_csv(tmp_path / "test.csv", index=False)

    frame = build_training_frame(tmp_path)
    assert len(frame) == 9
    assert (frame["source"] == "test_examples").sum() == 8


def test_training_frame_cleans_invalid_and_duplicate_rows(tmp_path: Path) -> None:
    pd.DataFrame([
        {"body": " valid\r\n", "rule": " rule ", "rule_violation": 1},
        {"body": " valid\r\n", "rule": " rule ", "rule_violation": 1},
        {"body": " ", "rule": "rule", "rule_violation": 0},
        {"body": "bad label", "rule": "rule", "rule_violation": 2},
    ]).to_csv(tmp_path / "train.csv", index=False)
    pd.DataFrame([{
        "row_id": 1,
        "body": "test",
        "positive_example_1": " p1 ",
        "positive_example_2": None,
        "negative_example_1": "n1\x00",
        "negative_example_2": " ",
        "rule": " rule ",
    }]).to_csv(tmp_path / "test.csv", index=False)

    frame = build_training_frame(tmp_path, example_repeats=1)
    assert len(frame) == 3
    assert set(frame["body"]) == {"valid", "p1", "n1"}
    assert set(frame["rule"]) == {"rule"}
    assert set(frame["rule_violation"]) == {0, 1}


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


def test_verified_submission_contract(tmp_path: Path) -> None:
    pd.DataFrame({"row_id": [1, 2], "rule_violation": [0.0, 0.0]}).to_csv(
        tmp_path / "sample_submission.csv", index=False
    )
    output = tmp_path / "submission7.csv"
    pd.DataFrame({"row_id": [2, 1], "rule_violation": [0.8, 0.2]}).to_csv(
        output, index=False
    )
    result = validate_submission(output, tmp_path)
    assert len(result) == 2
    assert result["row_id"].tolist() == [1, 2]
    assert pd.read_csv(output)["row_id"].tolist() == [1, 2]
