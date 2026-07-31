from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def describe_data(data_dir: Path) -> None:
    for name in ["train.csv", "test.csv", "sample_submission.csv"]:
        path = data_dir / name
        df = pd.read_csv(path)
        print(f"\n### {name}: shape={df.shape}")
        print(df.dtypes.to_string())
        print(df.head(3).to_string(max_colwidth=100))

    train = pd.read_csv(data_dir / "train.csv")
    print("\n### Target distribution")
    print(train["rule_violation"].value_counts(normalize=False).to_string())
    print((train["rule_violation"].value_counts(normalize=True) * 100).round(2).to_string())

    print("\n### Rules")
    print(train["rule"].value_counts().to_string())

    print("\n### Top subreddits")
    print(train["subreddit"].value_counts().head(20).to_string())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    describe_data(args.data_dir)
