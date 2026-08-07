"""生成项目根目录中的完整训练方法 Notebook。"""

from pathlib import Path

import nbformat as nbf


def markdown(text: str):
    return nbf.v4.new_markdown_cell(text.strip())


def code(text: str):
    return nbf.v4.new_code_cell(text.strip())


nb = nbf.v4.new_notebook()
nb["metadata"]["kernelspec"] = {
    "display_name": "Python 3",
    "language": "python",
    "name": "python3",
}
nb["metadata"]["language_info"] = {"name": "python", "version": "3.10"}
nb["cells"] = [
    markdown(
        """
# Jigsaw 规则违规分类：完整训练方法

这份 Notebook 是项目的主方法文档，同时也是一个可以在普通 Jupyter、VS Code Notebook、
Kaggle Notebook 或远程 GPU 服务器中执行的训练入口。它不依赖 Google Colab，也不包含
任何平台专属 API。

Notebook 完整覆盖：

1. 任务与数据字段；
2. 数据校验和保守清洗；
3. 将测试集正负示例转换成目标域监督样本；
4. 生成式模型的聊天 SFT 数据；
5. Ettin 编码器的二分类训练；
6. 推理、规则内排名与 submission 校验；
7. 可选多模型融合和实验记录。

默认不会启动昂贵训练。阅读和数据检查可以在 CPU 环境完成；训练 Ettin 或大语言模型时
需要 CUDA GPU。
"""
    ),
    markdown(
        """
## 1. 整体方法

```text
train.csv ───────────────────────────────┐
                                        │
test.csv 中的 positive/negative examples ├─→ 清洗、去重、目标域增强
                                        │
                                        └─→ 统一监督表
                                               │
                         ┌─────────────────────┴─────────────────────┐
                         │                                           │
                 生成式模型路线                               编码器模型路线
              body/rule → Yes/No SFT                    rule [SEP] body → BCE
                         │                                           │
                         └─────────────────────┬─────────────────────┘
                                               ↓
                                        原始违规分数
                                               ↓
                                      每条规则内部排名
                                               ↓
                                       submission.csv
```

任务的核心不是记住固定规则类别，而是学习评论内容与自然语言规则之间的匹配关系。测试集
提供的正负示例具有明确语义，可以作为新规则的目标域监督数据，但待预测评论本身没有标签，
不能加入训练。
"""
    ),
    markdown(
        """
## 2. 环境准备

在仓库根目录创建环境并安装：

```bash
python -m pip install -r requirements.txt
python -m pip install -e .
```

GPU 训练 Ettin 时还需要 `requirements-verified-colab.txt` 中的依赖。该文件虽然沿用历史
名称，但内容是普通 Python/Hugging Face 依赖，不要求 Colab：

```bash
python -m pip install -r requirements-verified-colab.txt
```

数据默认放在 `data/raw/`。可以使用 Kaggle CLI 下载，凭据通过本机环境或 Kaggle 配置文件
提供，不要写进 Notebook：

```bash
kaggle competitions download -c jigsaw-agile-community-rules -p data/raw
```
"""
    ),
    code(
        """
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def find_repo_root(start: Path) -> Path:
    \"\"\"从当前目录向上查找 pyproject.toml，定位仓库根目录。\"\"\"
    start = start.resolve()
    for candidate in (start, *start.parents):
        if (candidate / "pyproject.toml").exists():
            return candidate
    raise FileNotFoundError("没有找到仓库根目录，请从项目目录或其子目录启动 Notebook。")


REPO_ROOT = find_repo_root(Path.cwd())
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

DATA_DIR = REPO_ROOT / "data" / "raw"
OUTPUT_DIR = REPO_ROOT / "outputs" / "notebook_ettin"

print("仓库目录:", REPO_ROOT)
print("数据目录:", DATA_DIR)
print("输出目录:", OUTPUT_DIR)
"""
    ),
    markdown(
        """
## 3. 运行配置

- `EXAMPLE_REPEATS=2`：每条唯一测试正负示例初始出现一次，再追加一次。
- `RUN_SMOKE_TRAINING`：使用少量平衡样本验证下载、训练和推理闭环。
- `RUN_FULL_TRAINING`：使用构造后的完整训练集训练 Ettin。
- 两个训练开关默认均为 `False`，防止阅读 Notebook 时意外占用 GPU。
"""
    ),
    code(
        """
SEED = 3001
EXAMPLE_REPEATS = 2
SMOKE_ROWS = 32

MODEL_ID = "jhu-clsp/ettin-encoder-400m"
MODEL_REVISION = "7662476d60abb071a5bd319c9f3074f3072c062d"

RUN_SMOKE_TRAINING = False
RUN_FULL_TRAINING = False

required_files = {"train.csv", "test.csv", "sample_submission.csv"}
available_files = {path.name for path in DATA_DIR.glob("*.csv")}
DATA_READY = required_files <= available_files

print("数据是否就绪:", DATA_READY)
print("模型:", f"{MODEL_ID}@{MODEL_REVISION[:8]}")
if not DATA_READY:
    print("缺少文件:", sorted(required_files - available_files))
"""
    ),
    markdown(
        """
## 4. 原始数据字段

`train.csv` 的监督训练必需字段：

| 字段 | 含义 |
|---|---|
| `body` | Reddit 评论正文 |
| `rule` | 本条评论需要遵守的自然语言规则 |
| `rule_violation` | 二元标签，`1` 表示违规，`0` 表示不违规 |

`test.csv` 的关键字段：

| 字段 | 含义 |
|---|---|
| `row_id` | 提交时使用的唯一标识 |
| `body` | 待预测评论，不能作为有标签训练行 |
| `rule` | 需要判断的规则 |
| `positive_example_1/2` | 明确违反该规则的示例 |
| `negative_example_1/2` | 明确不违反该规则的示例 |
"""
    ),
    code(
        """
if DATA_READY:
    train_raw = pd.read_csv(DATA_DIR / "train.csv")
    test_raw = pd.read_csv(DATA_DIR / "test.csv")
    sample_submission = pd.read_csv(DATA_DIR / "sample_submission.csv")

    print("train:", train_raw.shape)
    print("test:", test_raw.shape)
    print("sample_submission:", sample_submission.shape)
    display(train_raw.head(3))
    display(test_raw.head(3))
else:
    print("数据尚未下载，本节跳过。")
"""
    ),
    markdown(
        """
## 5. 干净训练数据的构造

构造顺序是：

1. 校验必要列；
2. 统一换行、删除空字符和首尾空白；
3. 删除空评论、空规则和非法标签；
4. 原训练数据标记为 `source="train"`；
5. 测试正例转换成标签 `1`，负例转换成标签 `0`；
6. 测试示例标记为 `source="test_examples"`；
7. 在各来源内按 `body/rule/rule_violation/source` 去重；
8. 只重复测试示例，提高新规则监督信号的权重；
9. 使用固定随机种子打乱。

不会删除 URL、标点、emoji、大小写或评论内部换行，因为这些信息可能直接影响违规判断。
"""
    ),
    code(
        """
from first_place.data import build_training_frame

if DATA_READY:
    training_frame = build_training_frame(
        DATA_DIR,
        example_repeats=EXAMPLE_REPEATS,
        seed=SEED,
    )

    summary = {
        "rows": len(training_frame),
        "source_counts": training_frame["source"].value_counts().to_dict(),
        "label_counts": training_frame["rule_violation"].value_counts().to_dict(),
        "empty_body": int(training_frame["body"].str.len().eq(0).sum()),
        "empty_rule": int(training_frame["rule"].str.len().eq(0).sum()),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    display(training_frame.head(5))
else:
    training_frame = None
    print("数据尚未下载，本节跳过。")
"""
    ),
    markdown(
        """
### 测试示例如何展开

一条测试记录：

```json
{
  "body": "待预测评论",
  "rule": "No Advertising",
  "positive_example_1": "Buy my course at this link.",
  "negative_example_1": "I disliked the course."
}
```

转换后产生独立监督行：

```json
{"body": "Buy my course at this link.", "rule": "No Advertising", "rule_violation": 1}
{"body": "I disliked the course.", "rule": "No Advertising", "rule_violation": 0}
```

`body="待预测评论"` 没有真实标签，因此不会进入训练表。
"""
    ),
    code(
        """
if training_frame is not None:
    for source_name, group in training_frame.groupby("source"):
        print(f"\\n来源: {source_name}，行数: {len(group)}")
        display(group[["body", "rule", "rule_violation"]].head(3))
"""
    ),
    markdown(
        """
## 6. 路线 A：生成式模型聊天 SFT

生成式模型把二分类标签改写成一个极短的监督回答：

```text
rule_violation = 1 → "Yes"
rule_violation = 0 → "No"
```

每条训练记录转换为三轮消息：

```text
system: Reddit moderation ... Answer 'Yes' or 'No' only.
user:   Comment: {body}\\n\\nrule: {rule}
assistant: Yes 或 No
```

训练时应使用模型自己的 chat template，并且只对 assistant 回答区域计算交叉熵损失。
system 和 user token 只提供上下文，其 label 应 mask 为 `-100`。
"""
    ),
    code(
        """
SYSTEM_PROMPT = (
    "Reddit moderation: Does the comment violate the rule? "
    "Answer 'Yes' or 'No' only."
)


def build_sft_conversation(row: pd.Series) -> list[dict[str, str]]:
    completion = "Yes" if int(row["rule_violation"]) == 1 else "No"
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": f'Comment: {row["body"]}\\n\\nrule: {row["rule"]}',
        },
        {"role": "assistant", "content": completion},
    ]


if training_frame is not None:
    sft_example = build_sft_conversation(training_frame.iloc[0])
    print(json.dumps(sft_example, ensure_ascii=False, indent=2))
"""
    ),
    markdown(
        """
### SFT 配置

可使用 Qwen/Llama 等支持聊天模板的 Hugging Face 模型，并通过 4-bit LoRA 控制显存：

| 参数 | 建议值 |
|---|---|
| 量化 | 4-bit |
| LoRA rank | 16 |
| LoRA alpha | 32 |
| LoRA dropout | 0 |
| 目标模块 | attention 的 q/k/v/o 与 MLP 的 gate/up/down |
| 训练长度 | 256 |
| 推理长度 | 512 |
| Epoch | 1 |
| Learning rate | 1.5e-4 |
| Optimizer | AdamW 8-bit |
| Loss | 只计算 assistant 的 Yes/No token |

更换模型时必须重新检查：

- chat template 的 system/user/assistant 特殊 token；
- `Yes`、`No` 是否被拆成多个 token；
- response-only mask 是否正确覆盖回答区；
- tokenizer 的左截断和 padding 设置。
"""
    ),
    code(
        """
SFT_CONFIG = {
    "load_in_4bit": True,
    "lora_r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.0,
    "target_modules": [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    "max_train_length": 256,
    "max_inference_length": 512,
    "epochs": 1,
    "learning_rate": 1.5e-4,
    "weight_decay": 0.01,
    "packing": False,
    "response_only_loss": True,
}

print(json.dumps(SFT_CONFIG, ensure_ascii=False, indent=2))
"""
    ),
    markdown(
        """
### 生成式模型的分类分数

推理不需要生成长文本。对 assistant 第一个输出位置取 vocabulary logits，分别聚合肯定
和否定候选 token：

```text
肯定：Yes, YES, Y, yes, True
否定：No, NO, N, no, False
```

在两组聚合 log-probability 之间做 softmax，即可得到 `p_yes`。这种做法把生成式模型
转换成稳定的二分类打分器。
"""
    ),
    code(
        """
def binary_probability_from_logps(
    yes_logp: np.ndarray,
    no_logp: np.ndarray,
) -> np.ndarray:
    \"\"\"将 Yes/No 的对数分数转换为二分类概率。\"\"\"
    pair = np.column_stack([yes_logp, no_logp])
    pair = pair - pair.max(axis=1, keepdims=True)
    probs = np.exp(pair)
    probs = probs / probs.sum(axis=1, keepdims=True)
    return probs[:, 0]


demo = binary_probability_from_logps(
    np.array([-0.2, -2.0]),
    np.array([-1.8, -0.1]),
)
print("示例 p_yes:", demo)
"""
    ),
    markdown(
        """
## 7. 路线 B：Ettin 编码器分类

项目的默认可运行路线使用 `jhu-clsp/ettin-encoder-400m`。输入格式是：

```text
rule + tokenizer.sep_token + body
```

模型使用单输出 sequence classification head：

```text
文本 → Ettin Encoder → 单个 logit → BCEWithLogitsLoss
```

默认参数：

- 最大长度 512；
- batch size 8；
- gradient accumulation 2；
- 1 epoch；
- AdamW，learning rate `2e-5`；
- cosine learning-rate schedule；
- FP16 自动混合精度；
- gradient clipping 1.0。
"""
    ),
    code(
        """
def build_encoder_text(rule: str, body: str, sep_token: str = "[SEP]") -> str:
    return f"{rule}{sep_token}{body}"


if training_frame is not None:
    encoder_preview = training_frame.head(3).copy()
    encoder_preview["text"] = encoder_preview.apply(
        lambda row: build_encoder_text(row["rule"], row["body"]),
        axis=1,
    )
    display(encoder_preview[["text", "rule_violation"]])
"""
    ),
    markdown(
        """
## 8. 执行真实 Ettin 训练

统一 runner 会完成：

1. 从 Hugging Face 下载固定 revision；
2. 将模型、数据和输出路径传给训练子进程；
3. 训练后立即推理；
4. 校验 submission 的列、行数、row_id、有限值和范围；
5. 写出模型、日志和 `run_manifest.json`。

smoke 模式会使用每类少量样本并跳过模型保存，适合先验证环境。完整训练只应在 smoke
成功后开启。
"""
    ),
    code(
        """
runner = REPO_ROOT / "scripts" / "run_verified_ettin.py"

if RUN_SMOKE_TRAINING:
    subprocess.run(
        [
            sys.executable,
            str(runner),
            "--data-dir",
            str(DATA_DIR),
            "--output-dir",
            str(OUTPUT_DIR / "smoke"),
            "--smoke-rows",
            str(SMOKE_ROWS),
        ],
        check=True,
    )
else:
    print("RUN_SMOKE_TRAINING=False，未启动 smoke 训练。")

if RUN_FULL_TRAINING:
    subprocess.run(
        [
            sys.executable,
            str(runner),
            "--data-dir",
            str(DATA_DIR),
            "--output-dir",
            str(OUTPUT_DIR / "full"),
        ],
        check=True,
    )
else:
    print("RUN_FULL_TRAINING=False，未启动完整训练。")
"""
    ),
    markdown(
        """
## 9. 规则内排名

不同规则的样本难度和原始概率尺度可能不同。项目在每条规则内把分数转换为排名：

```text
(rank - 1) / max(group_size - 1, 1)
```

最低分映射为 0，最高分映射为 1。ROC AUC 主要关注排序，因此该步骤可以减少不同规则
之间概率尺度不一致带来的影响。
"""
    ),
    code(
        """
def rank_within_rule(
    frame: pd.DataFrame,
    score_column: str = "raw_score",
) -> pd.Series:
    ranks = frame.groupby("rule")[score_column].rank(
        method="average",
        ascending=True,
    )
    sizes = frame.groupby("rule")[score_column].transform("size")
    denominator = (sizes - 1).where(sizes > 1, 1)
    return (ranks - 1) / denominator


rank_demo = pd.DataFrame(
    {
        "rule": ["A", "A", "A", "B", "B"],
        "raw_score": [0.4, 0.1, 0.8, 0.7, 0.2],
    }
)
rank_demo["rule_violation"] = rank_within_rule(rank_demo)
display(rank_demo)
"""
    ),
    markdown(
        """
## 10. Submission 校验

最终文件必须满足：

- 列严格为 `row_id, rule_violation`；
- 行数与 `sample_submission.csv` 相同；
- row_id 集合完全一致且没有重复；
- 分数为有限数值；
- 分数位于 `[0, 1]`；
- 输出顺序恢复为 sample submission 的顺序。
"""
    ),
    code(
        """
from first_place.validation import validate_submission

candidate_paths = [
    OUTPUT_DIR / "full" / "submission7.csv",
    OUTPUT_DIR / "smoke" / "submission7.csv",
]
existing_submission = next((path for path in candidate_paths if path.exists()), None)

if DATA_READY and existing_submission is not None:
    checked = validate_submission(existing_submission, DATA_DIR)
    print("校验通过:", existing_submission)
    display(checked.head())
else:
    print("尚无预测文件，本节只展示校验契约。")
"""
    ),
    markdown(
        """
## 11. 可选多模型融合

单模型足以验证完整工程闭环。如果后续训练了多个模型，可按 `row_id` 对齐预测，再进行
加权平均。融合时不能依赖 CSV 的当前行顺序，并且只使用已经存在且通过校验的预测文件。

编码器和生成式模型结构差异较大，错误模式也不同，因此它们的融合通常比相同架构的简单
重复更有价值。权重应通过同一验证方案选择，不应直接比较不同数据切分或不同指标的分数。
"""
    ),
    code(
        """
from first_place.ensemble import blend

# 示例：只有在对应预测文件已经存在时才执行。
RUN_ENSEMBLE = False
if RUN_ENSEMBLE:
    blended = blend(
        input_dir=OUTPUT_DIR / "predictions",
        output_path=OUTPUT_DIR / "submission_blend.csv",
        weights={
            "submission1.csv": 0.5,
            "submission7.csv": 0.5,
        },
    )
    display(blended.head())
else:
    print("RUN_ENSEMBLE=False，当前使用单模型路线。")
"""
    ),
    markdown(
        """
## 12. 实验记录与完成标准

每次训练至少记录：

- Git commit；
- 数据行数、来源分布和标签分布；
- Hugging Face model ID 与 revision；
- 随机种子；
- 最大长度、batch size、梯度累积和 epoch；
- smoke/full 标记；
- 模型、日志、manifest 和 submission 路径；
- 使用的数据切分、指标和最终分数。

一个完整运行应满足：

1. 数据构造无空文本和非法标签；
2. 包导入与单元测试通过；
3. smoke 训练和推理成功；
4. submission 契约校验通过；
5. 完整训练产物写入持久化目录；
6. 只有在相同数据、切分和指标下比较实验结果。
"""
    ),
    code(
        """
notebook_manifest = {
    "repo_root": str(REPO_ROOT),
    "data_ready": DATA_READY,
    "model_id": MODEL_ID,
    "model_revision": MODEL_REVISION,
    "seed": SEED,
    "example_repeats": EXAMPLE_REPEATS,
    "run_smoke_training": RUN_SMOKE_TRAINING,
    "run_full_training": RUN_FULL_TRAINING,
}
print(json.dumps(notebook_manifest, ensure_ascii=False, indent=2))
"""
    ),
]

# nbformat 默认随机生成 cell id；固定 ID 可确保重复构建不会产生无意义的 Git diff。
for index, cell in enumerate(nb["cells"]):
    cell["id"] = f"jigsaw-{index:03d}"

target = Path("jigsaw_training_method.ipynb")
nbf.write(nb, target)
print(target)
