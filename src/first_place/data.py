from __future__ import annotations

from pathlib import Path

import pandas as pd


PROMPT = "Reddit moderation: Does the comment violate the rule? Answer 'Yes' or 'No' only."
# 训练和测试文件必须具备的最小字段集合；尽早校验可以避免训练中途才发现数据问题。
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
    """保守清洗文本：只处理格式噪声，不改变评论或规则的语义。"""
    return (
        series.astype("string")
        .str.replace("\r\n", "\n", regex=False)
        .str.replace("\r", "\n", regex=False)
        .str.replace("\x00", "", regex=False)
        .str.strip()
    )


def _clean_labeled_rows(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    # 评论和规则使用相同的清洗规则，保证训练与推理的输入格式一致。
    for column in TEXT_COLUMNS:
        result[column] = _clean_text(result[column])
    # 无法解析的标签转成 NaN，随后与空文本一起过滤。
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
    """将原训练行和测试集正负示例整理成统一的带标签训练表。

    ``example_repeats=2`` 表示每条唯一测试示例初始出现一次，再追加一次。
    调整该参数可以控制目标规则示例在训练集中的权重。
    """
    if example_repeats < 1:
        raise ValueError("example_repeats must be at least 1")

    train = pd.read_csv(data_dir / "train.csv")
    test = pd.read_csv(data_dir / "test.csv")
    _require_columns(train, TRAIN_COLUMNS, "train.csv")
    _require_columns(test, TEST_COLUMNS, "test.csv")

    # 原训练集只保留分类必需字段，source 用于后续统计和分来源去重。
    base = _clean_labeled_rows(
        train[["body", "rule", "rule_violation"]].assign(source="train")
    )

    # 测试集提供的正负示例具有明确语义，可展开成独立监督样本：
    # positive -> 1，negative -> 0。待预测的 test.body 不参与训练。
    examples = []
    for polarity, label in (("positive", 1), ("negative", 0)):
        for number in (1, 2):
            column = f"{polarity}_example_{number}"
            frame = test[[column, "rule"]].rename(columns={column: "body"})
            examples.append(frame.assign(rule_violation=label, source="test_examples"))

    example_frame = _clean_labeled_rows(pd.concat(examples, ignore_index=True))
    # 先去重再重复采样，否则重复行会被后续去重抵消。
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
        # 只提高目标规则示例的权重，不重复原训练集。
        result = pd.concat(
            [result, *([example_frame] * (example_repeats - 1))],
            ignore_index=True,
        )
    # 固定种子打乱，确保相同数据与配置下样本顺序可重复。
    return result.sample(frac=1, random_state=seed).reset_index(drop=True)


def build_classification_text(frame: pd.DataFrame) -> pd.Series:
    return frame.apply(lambda row: build_prompt(str(row["body"]), str(row["rule"])), axis=1)
