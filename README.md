# Jigsaw Agile Community Rules：高分方案单模型复现

本项目以竞赛第一名 notebook `1st-place-code (1).ipynb` 为参考，保留其核心思路，
并提供一条适合 Google Colab 验证的单模型闭环：

```text
GitHub 代码
  → Colab 干净运行时克隆仓库
  → Colab Secrets 获取 Kaggle/Hugging Face 凭据
  → 下载竞赛数据
  → 从 Hugging Face 下载固定版本的 Ettin-400M
  → 构造干净训练数据
  → GPU 微调与推理
  → 校验 submission
  → 模型、日志和预测保存到 Google Drive
```

[在 Colab 中运行已验证的 Ettin 单模型闭环](https://colab.research.google.com/github/mingzhuoFUN/jigsaw-agile-community-rules/blob/agent/winner-style-colab-pipeline/notebooks/verified_ettin_colab.ipynb)

多模型代码和原始 notebook 仍保留在仓库中，用于学习第一名方案；但证明工程链路可运行
不要求在 Colab 中重复训练全部 14B/8B/4B 模型。

| 项目 | 内容 |
|---|---|
| 任务 | 判断 Reddit 评论是否违反给定社区规则 |
| 输入 | `body`、`subreddit`、`rule`、正例与反例 |
| 输出 | 每个 `row_id` 的 `rule_violation` 概率 |
| 评估指标 | ROC AUC |
| 核心难点 | 测试集包含训练阶段未出现的规则，需要基于规则语义泛化 |
| 推荐入口 | [在 Google Colab 中运行 Ettin 模型](https://colab.research.google.com/github/mingzhuoFUN/jigsaw-agile-community-rules/blob/main/notebooks/verified_ettin_colab.ipynb) |

输入包含：

- `body`：待审核的 Reddit 评论；
- `subreddit`：评论所在社区；
- `rule`：需要判断是否违反的社区规则；
- `positive_example_1/2`：违反该规则的示例；
- `negative_example_1/2`：不违反该规则的示例。

目标是为每个 `row_id` 预测 `rule_violation` 分数，竞赛指标为 AUC。

训练集只提供少量规则，而测试集可能包含训练时没有出现过的规则。因此模型不能只记忆
固定规则类别，而必须学习“规则文本与评论内容是否匹配”。测试集中的正负示例是适应
新规则的重要监督信息。

## 第一名方案的核心思路

参考 notebook 使用多个生成式 LLM 与一个编码器模型进行融合：

1. 把原始训练集作为基础监督样本。
2. 把测试集给出的正例标记为 `1`、负例标记为 `0`。
3. 对这些目标规则示例进行重复采样，使模型优先学习测试域中的新规则。
4. 生成式模型学习回答 `Yes/No`；Ettin 编码器直接学习二分类。
5. 推理后在每条规则内部进行 rank normalization。
6. 融合多个结构、参数规模和随机种子的模型。

本仓库的 Colab 主入口选择其中的 `jhu-clsp/ettin-encoder-400m`。这样保留了高分方案的
规则感知数据构造、目标域示例增强、分类训练和按规则排序，同时把验证成本控制在单 GPU
可接受范围内。

## 如何构造干净训练数据

统一的数据实现位于
[`src/first_place/data.py`](src/first_place/data.py)，训练脚本和测试都调用同一个函数，
避免 notebook、README 和实际训练出现不同语义。

### 1. 校验数据结构

`train.csv` 至少需要：

```text
body, rule, rule_violation
```

`test.csv` 至少需要：

```text
row_id, body, rule,
positive_example_1, positive_example_2,
negative_example_1, negative_example_2
```

缺少必要列时立即报错，而不是训练到一半才失败。

### 2. 保守清洗文本

评论和规则只进行不会明显改变语义的清洗：

- 使用 Pandas nullable string 类型；
- 把 Windows/旧式换行统一为 `\n`；
- 删除非法空字符 `\x00`；
- 删除文本首尾空白；
- 删除缺失或清洗后为空的评论/规则。

不会删除标点、URL、大小写、emoji、Markdown 或评论内部换行，因为这些内容可能是判断
广告、攻击或其他违规行为的重要信号。

### 3. 清洗标签

- 把标签转换为数值；
- 删除不能转换的标签；
- 只保留 `0` 和 `1`；
- 最终保存为整数标签。

### 4. 构造目标规则示例

对测试集的四个示例列执行：

```text
positive_example_1/2 → body，rule_violation = 1
negative_example_1/2 → body，rule_violation = 0
```

这些行标记为 `source="test_examples"`，原始训练行标记为 `source="train"`，方便记录
数据来源和检查增强比例。

### 5. 去重与重复采样

先在各来源内按以下字段去重：

```text
body, rule, rule_violation, source
```

然后重复目标规则示例。默认 `example_repeats=2`，这与第一名 notebook 的可执行代码一致：
每条唯一测试示例初始出现一次，再追加一次。

原 notebook 的注释声称总共出现 3 次，但代码实际只产生 2 次。本仓库以真实执行行为为
忠实复现默认值；如果需要研究 3 次重复，可以显式设置 `example_repeats=3`，但应把它
视为独立实验。

### 6. 固定随机打乱

数据构造使用固定随机种子，保证相同数据和代码版本能够得到相同的样本顺序。

本地检查构造结果：

```powershell
pip install -e .
python -c "from pathlib import Path; from first_place.data import build_training_frame; print(build_training_frame(Path('data/raw')).groupby(['source','rule_violation']).size())"
```

## Ettin 单模型训练方法

训练脚本位于
[`scripts/winner/train_ettin_400m.py`](scripts/winner/train_ettin_400m.py)。

### 输入表示

每条样本构造成：

```text
rule + tokenizer.sep_token + body
```

规则放在评论前面，使模型首先获得判断标准，再编码待审核内容。

### 模型和损失

- 基础模型：`jhu-clsp/ettin-encoder-400m`；
- 任务头：单输出 sequence classification；
- 损失：`BCEWithLogitsLoss`；
- 优化器：AdamW；
- 学习率调度：cosine；
- 默认训练：1 epoch；
- 默认最大长度：512；
- GPU 环境使用 FP16；
- 固定模型 revision，避免 Hugging Face 仓库更新改变结果。

### 推理和提交分数

模型先输出 sigmoid 分数，然后在每条 `rule` 内部进行排名：

```text
(rank - 1) / max(group_size - 1, 1)
```

这种处理保留同一规则内样本的相对次序，减少不同规则之间概率尺度不一致的影响。最终输出：

```text
row_id, rule_violation
```

runner 会验证：

- 列名完全正确；
- 行数和 `sample_submission.csv` 一致；
- `row_id` 集合一致且无重复；
- 预测无缺失；
- 所有分数均在 `[0, 1]`。

## 在 Colab 中运行

### 准备

1. 打开上面的 Colab 链接。
2. 选择 GPU runtime。
3. 在 Colab Secrets 添加 `KAGGLE_API_TOKEN`。
4. Kaggle 账号必须已经接受比赛规则。
5. 可选：添加只读 `HF_TOKEN`，减少匿名下载限流。

不要把 token 写入 notebook、代码单元格或 GitHub。

### 执行阶段

Notebook 会依次：

1. 从 GitHub 克隆指定分支的全新工作区；
2. 检查 CUDA；
3. 安装固定的关键依赖和当前仓库包；
4. 编译代码并运行单元测试；
5. 下载 Kaggle 数据；
6. 展示构造后的数据规模、来源和标签分布；
7. 从 Hugging Face 下载固定 revision 的模型；
8. 先执行 32 行真实训练 smoke test；
9. 再执行完整训练；
10. 将结果保存到 Google Drive。

Google Drive 输出目录默认是：

```text
MyDrive/jigsaw-verified-ettin/
```

其中包含：

- `model/`：微调后的模型和 tokenizer；
- `training.log`：训练日志；
- `run_manifest.json`：GitHub/Hugging Face 复现信息和数据规模；
- `submission7.csv`：经过校验的提交文件。

如果只想验证链路，可把 notebook 中的 `RUN_FULL_TRAINING` 改为 `False`。32 行 smoke test
仍然会真实下载模型、训练并推理，因此比只检查 import 更有意义。

## 本地开发

安装：

```powershell
pip install -r requirements.txt
pip install -e .
```

运行测试：

```powershell
python -m pytest -q
```

运行传统 TF-IDF baseline：

```powershell
python src/train_baseline.py
```

本地或 GPU 环境运行 Ettin：

```powershell
python scripts/run_verified_ettin.py `
  --data-dir data/raw `
  --output-dir outputs/ettin
```

低成本 smoke：

```powershell
python scripts/run_verified_ettin.py `
  --data-dir data/raw `
  --output-dir outputs/ettin-smoke `
  --smoke-rows 32
```

## 仓库结构

```text
1st-place-code (1).ipynb          第一名原始 notebook
reference/notebook_cells/         原 notebook 脚本冻结副本
scripts/winner/                   适配在线模型和可移植路径的训练脚本
scripts/run_verified_ettin.py     单模型闭环 runner
notebooks/verified_ettin_colab.ipynb
src/first_place/data.py           统一数据清洗和构造
src/first_place/ensemble.py       多模型预测融合参考
tests/                            数据、提交和融合回归测试
REPRODUCTION.md                   原始方案与适配差异
```

## 复现边界

- 单模型 Colab 用于证明完整工程和训练思路可以运行，不声称等同于第一名多模型成绩。
- 第一名 notebook 没有提供可直接复核的本地 OOF 或 leaderboard 分数。
- 只有在相同数据、指标和竞赛环境中提交后，才能进行可靠分数比较。
- `reference/notebook_cells/` 保持冻结；环境适配和改进只发生在 runnable 代码中。
