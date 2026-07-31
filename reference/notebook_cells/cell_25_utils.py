import pandas as pd
from datasets import Dataset
from constants import POSITIVE_ANSWER, NEGATIVE_ANSWER, COMPLETE_PHRASE, BASE_PROMPT

def build_prompt(row):
    # Kept for parity; not used by the DeBERTa pipeline
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

    # upsample target block (test examples) by labeling them now
    for violation_type in ["positive", "negative"]:
        for i in range(1, 3):
            col = f"{violation_type}_example_{i}"
            sub_dataset = test_dataset[[col, "rule"]].copy()
            sub_dataset = sub_dataset.rename(columns={col: "body"})
            sub_dataset["rule_violation"] = 1 if violation_type == "positive" else 0
            sub_dataset["source"] = "test_examples"
            flatten.append(sub_dataset)

    dataframe = pd.concat(flatten, axis=0, ignore_index=True)
    dataframe = dataframe.drop_duplicates(ignore_index=True)

    # upsample test_examples once more (2x extra copies -> appears 3x total)
    test_rows = dataframe[dataframe["source"] == "test_examples"]
    if not test_rows.empty:
        dataframe = pd.concat([dataframe, test_rows], axis=0, ignore_index=True)

    dataframe = dataframe.sample(frac=1.0, random_state=3001).reset_index(drop=True)
    dataframe = dataframe.drop(columns=["source"])
    return dataframe

def build_classification_dataframe(df, tok_sep_token="</s>"):
    """
    Build a dataframe for classification using text = rule + [SEP] + comment.
    """
    df = df.copy()
    sep = tok_sep_token if tok_sep_token else "</s>"
    df["text"] = df["rule"].astype(str) + sep + df["body"].astype(str)
    cols = ["text"]
    if "rule_violation" in df:
        cols.append("rule_violation")
    return df[cols]

def hf_dataset_from_dataframe(df):
    return Dataset.from_pandas(df.reset_index(drop=True))