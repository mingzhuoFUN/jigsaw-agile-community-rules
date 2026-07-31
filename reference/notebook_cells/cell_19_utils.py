import pandas as pd
from datasets import Dataset
from constants import POSITIVE_ANSWER, NEGATIVE_ANSWER, COMPLETE_PHRASE, BASE_PROMPT


def build_prompt(row):
    return f"""
{BASE_PROMPT}

Comment: {row["body"]}

rule: {row["rule"]}
---
{COMPLETE_PHRASE}"""


def get_dataframe_to_train(data_path):
    train_dataset = pd.read_csv(f"{data_path}/train.csv")
    test_dataset = pd.read_csv(f"{data_path}/test.csv")

    flatten = []

    # base train rows
    base = train_dataset[["body", "rule", "rule_violation"]].copy()
    base["source"] = "train"
    flatten.append(base)

    # >>> Upsample target block (test examples) later by labeling them now
    for violation_type in ["positive", "negative"]:
        for i in range(1, 3):
            col = f"{violation_type}_example_{i}"
            sub_dataset = test_dataset[[col, "rule"]].copy()
            sub_dataset = sub_dataset.rename(columns={col: "body"})
            sub_dataset["rule_violation"] = 1 if violation_type == "positive" else 0
            sub_dataset["source"] = "test_examples"
            flatten.append(sub_dataset)

    # combine & dedupe first (so oversampling isn't undone)
    dataframe = pd.concat(flatten, axis=0, ignore_index=True)
    dataframe = dataframe.drop_duplicates(ignore_index=True)

    # upsample test_examples to appear 3x total (add two extra copies)
    test_rows = dataframe[dataframe["source"] == "test_examples"]
    if not test_rows.empty:
        dataframe = pd.concat([dataframe, test_rows], axis=0, ignore_index=True)

    # optional: shuffle for randomness
    dataframe = dataframe.sample(frac=1.0, random_state=2003).reset_index(drop=True)

    # drop helper column before returning
    dataframe = dataframe.drop(columns=["source"])

    return dataframe


def build_dataset(dataframe):
    dataframe["prompt"] = dataframe.apply(build_prompt, axis=1)

    columns = ["prompt"]
    if "rule_violation" in dataframe:
        dataframe["completion"] = dataframe["rule_violation"].map(
            {
                1: POSITIVE_ANSWER,
                0: NEGATIVE_ANSWER,
            }
        )
        columns.append("completion")

    dataframe = dataframe[columns]
    dataset = Dataset.from_pandas(dataframe)
    return dataset