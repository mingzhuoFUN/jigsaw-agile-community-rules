from __future__ import annotations

import pandas as pd


TEXT_COLUMNS = [
    "body",
    "rule",
    "subreddit",
    "positive_example_1",
    "positive_example_2",
    "negative_example_1",
    "negative_example_2",
]


def build_pair_text(df: pd.DataFrame) -> pd.Series:
    """Build a compact rule-aware text representation for vector models."""
    missing = [col for col in TEXT_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    filled = df[TEXT_COLUMNS].fillna("")
    return (
        "COMMENT: "
        + filled["body"]
        + "\nSUBREDDIT: "
        + filled["subreddit"]
        + "\nRULE: "
        + filled["rule"]
        + "\nVIOLATING EXAMPLES: "
        + filled["positive_example_1"]
        + " [SEP] "
        + filled["positive_example_2"]
        + "\nNON-VIOLATING EXAMPLES: "
        + filled["negative_example_1"]
        + " [SEP] "
        + filled["negative_example_2"]
    )
