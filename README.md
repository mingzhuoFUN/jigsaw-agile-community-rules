# Jigsaw Agile Community Rules：单模型训练

[阅读或运行完整训练方法 Notebook](https://github.com/mingzhuoFUN/jigsaw-agile-community-rules/blob/main/jigsaw_training_method.ipynb)

`jigsaw_training_method.ipynb` 位于仓库根目录，是项目的主方法文档。它完整展示数据构造、
生成式聊天 SFT、Ettin 编码器训练、推理打分、规则内排名和可选集成，可在普通 Jupyter、
VS Code Notebook、Kaggle Notebook 或 GPU 服务器中运行，不依赖 Google Colab。

```text
GitHub 代码
  → Jupyter / Kaggle / Colab / GPU 服务器
  → 本地环境或环境变量提供 Kaggle/Hugging Face 凭据
  → 下载竞赛数据
  → 从 Hugging Face 下载固定版本的 Ettin-400M
  → 构造干净训练数据
  → GPU 微调与推理
  → 校验 submission
  → 模型、日志和预测保存到指定输出目录
```

当前默认使用单个 Ettin-400M 模型验证完整训练链路。项目不要求训练多个 14B/8B/4B
模型，也可以在闭环跑通后按需要扩展其他 Hugging Face 模型。

| 项目 | 内容 |
|---|---|
| 任务 | 判断 Reddit 评论是否违反给定社区规则 |
| 输入 | `body`、`subreddit`、`rule`、正例与反例 |
| 输出 | 每个 `row_id` 的 `rule_violation` 概率 |
| 评估指标 | ROC AUC |
| 核心难点 | 测试集包含训练阶段未出现的规则，需要基于规则语义泛化 |
| 完整方法 | [打开独立 Jupyter Notebook](https://github.com/mingzhuoFUN/jigsaw-agile-community-rules/blob/main/jigsaw_training_method.ipynb) |
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

## 项目的核心训练思路

项目同时保留生成式模型和编码器模型两种训练路线：

1. 把原始训练集作为基础监督样本。
2. 把测试集给出的正例标记为 `1`、负例标记为 `0`。
3. 对这些目标规则示例进行重复采样，使模型优先学习测试域中的新规则。
4. 生成式模型学习回答 `Yes/No`；Ettin 编码器直接学习二分类。
5. 推理后在每条规则内部进行 rank normalization。
6. 融合多个结构、参数规模和随机种子的模型。

Colab 主入口选择 `jhu-clsp/ettin-encoder-400m`，以规则感知数据构造、目标域示例增强、
分类训练和按规则排序为核心，同时把验证成本控制在单 GPU 可接受范围内。

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

然后重复目标规则示例。默认 `example_repeats=2`：每条唯一测试示例初始出现一次，
再追加一次。这个参数用于提高目标规则示例在训练数据中的权重；如果需要研究不同增强
强度，可以显式设置 `example_repeats=1`、`3` 或其他正整数，并分别记录验证结果。

### 6. 固定随机打乱

数据构造使用固定随机种子，保证相同数据和代码版本能够得到相同的样本顺序。

本地检查构造结果：

```powershell
pip install -e .
python -c "from pathlib import Path; from first_place.data import build_training_frame; print(build_training_frame(Path('data/raw')).groupby(['source','rule_violation']).size())"
```

## 训练数据最终是什么格式

训练代码并不是直接把整行 CSV 交给模型。它先把不同来源的数据统一为一个最小监督格式，
再根据模型类型转换成聊天 SFT 数据或分类数据。

### 原始训练行

下面是结构示例，文本仅用于说明格式：

| body | rule | rule_violation | subreddit | positive_example_1 | negative_example_1 |
|---|---|---:|---|---|---|
| `Visit my shop and use code SAVE20.` | `No Advertising` | 1 | `example_forum` | `Buy this product here.` | `I bought this product yesterday.` |

虽然原始训练文件还包含 `subreddit` 和正负例字段，当前训练代码实际只选取：

```text
body, rule, rule_violation
```

得到的基础监督样本是：

```json
{
  "body": "Visit my shop and use code SAVE20.",
  "rule": "No Advertising",
  "rule_violation": 1,
  "source": "train"
}
```

`subreddit` 没有进入当前模型的最终训练文本，正负例也不会和当前评论一起拼成一个超长
prompt。它们会被拆成独立的带标签样本。

### 测试正负例如何变成训练行

假设测试集的一行是：

```json
{
  "row_id": 18,
  "body": "待预测的评论",
  "rule": "No Advertising",
  "positive_example_1": "Buy my course at this link.",
  "positive_example_2": "Use my referral code ABC.",
  "negative_example_1": "I disliked the course.",
  "negative_example_2": "Where can I read the rules?"
}
```

训练代码不会使用这行的 `body` 作为有标签训练数据，因为它的真实标签未知。它只展开
已经由比赛提供了语义极性的四个示例：

| body | rule | rule_violation | source |
|---|---|---:|---|
| `Buy my course at this link.` | `No Advertising` | 1 | `test_examples` |
| `Use my referral code ABC.` | `No Advertising` | 1 | `test_examples` |
| `I disliked the course.` | `No Advertising` | 0 | `test_examples` |
| `Where can I read the rules?` | `No Advertising` | 0 | `test_examples` |

因此，构造后的统一训练表只有四个字段：

```text
body: string
rule: string
rule_violation: 0 或 1
source: train 或 test_examples
```

`source` 只用于去重、重复采样和统计，送入模型前会删除。

### 完整构造顺序

对每份数据严格按以下顺序处理：

```text
train.csv
  └─ 选择 body/rule/rule_violation
  └─ source = "train"
                                      ┐
test.csv                              │
  ├─ positive_example_1 → label 1     │
  ├─ positive_example_2 → label 1     ├─ 合并
  ├─ negative_example_1 → label 0     │
  └─ negative_example_2 → label 0     │
     source = "test_examples"         ┘
          ↓
字段和标签校验
          ↓
保守文本清洗
          ↓
按 body/rule/label/source 去重
          ↓
重复 test_examples（默认总共出现 2 次）
          ↓
固定随机种子打乱
          ↓
统一监督训练表
```

这里利用了测试集公开示例，但没有使用待预测评论的未知标签。其意义是把比赛提供的
demonstrations 转换成目标域监督数据，让模型学到测试阶段的新规则。

本仓库当前数据经过处理后得到：

| 来源 | 行数 | 说明 |
|---|---:|---|
| `train` | 1884 | 原训练数据清洗、去重后的唯一行 |
| `test_examples` | 76 | 唯一正负示例按默认配置重复后的行 |
| 总计 | 1960 | 实际送入训练流程的行数 |

这些数字依赖当前竞赛文件；更换数据版本时，以 `run_manifest.json` 中记录的统计为准。

## 生成式模型如何进行 SFT

项目中的 Qwen、Llama 等生成式模型采用 Supervised Fine-Tuning。这里的 SFT 目标非常窄：
给模型一条评论和一条规则，只学习输出 `Yes` 或 `No`。

### 1. 标签转换为 completion

统一训练表中的二元标签被转换为文本答案：

```text
rule_violation = 1 → completion = "Yes"
rule_violation = 0 → completion = "No"
```

送入 SFT 构造器的表最终是：

```text
body, rule, completion
```

示例：

```json
{
  "body": "Visit my shop and use code SAVE20.",
  "rule": "No Advertising",
  "completion": "Yes"
}
```

### 2. 转换成三轮聊天消息

每条样本转换成一个 system/user/assistant conversation：

```json
[
  {
    "role": "system",
    "content": "Reddit moderation: Does the comment violate the rule? Answer 'Yes' or 'No' only."
  },
  {
    "role": "user",
    "content": "Comment: Visit my shop and use code SAVE20.\n\nrule: No Advertising"
  },
  {
    "role": "assistant",
    "content": "Yes"
  }
]
```

负样本的唯一结构差别是 assistant 内容为 `No`。

### 3. 使用模型原生 chat template 序列化

训练代码调用 tokenizer 的 `apply_chat_template`，把结构化消息转换成模型真正看到的
token 序列。以 Qwen 风格表示，逻辑结构大致是：

```text
<system>
Reddit moderation: Does the comment violate the rule?
Answer 'Yes' or 'No' only.
</system>
<user>
Comment: Visit my shop and use code SAVE20.

rule: No Advertising
</user>
<assistant>
Yes
</assistant>
```

实际特殊 token 由所选模型的 tokenizer 决定，不应手工把上面的展示标签写进数据。
Qwen3 路径还设置 `enable_thinking=False`，避免模型学习或生成思维链，只训练直接回答。

当前生成式训练代码在 chat template 输出后使用 `[:-11]` 去除末尾模板内容。这是针对
当前 tokenizer 输出的实现细节；如果以后更换 tokenizer，必须重新检查，不能假设固定
截取 11 个字符始终正确。

### 4. 只对 assistant 回答计算 loss

训练代码使用 `train_on_responses_only`。system prompt 和 user 中的评论、规则用于提供
上下文，但它们对应的 label 会被 mask 为 `-100`，不参与交叉熵损失。

可以把训练目标理解为：

```text
system tokens      → 只作为输入，不计算 loss
user tokens        → 只作为输入，不计算 loss
assistant prefix   → 定位回答区间
Yes/No tokens      → 计算 loss
```

因此模型优化的是：

```text
P("Yes" | system prompt, comment, rule)
P("No"  | system prompt, comment, rule)
```

而不是学习复述评论、规则或生成长篇解释。这也是这个 SFT 方案适合二分类比赛的原因。

### 5. LoRA 与 SFT 参数

以项目中的 Qwen3-14B 配置为例：

| 参数 | 设置 |
|---|---|
| 权重量化 | 4-bit，亦支持 8-bit |
| LoRA rank | `r=16` |
| LoRA alpha | `32` |
| LoRA dropout | `0.0` |
| LoRA bias | `none` |
| 目标模块 | `q/k/v/o_proj` 和 `gate/up/down_proj` |
| Gradient checkpointing | Unsloth 模式 |
| 训练最大长度 | 256 tokens |
| 推理最大长度 | 512 tokens |
| Packing | `False` |
| Epoch | 1 |
| Learning rate | `1.5e-4` |
| Weight decay | `0.01` |
| Scheduler | linear |
| Optimizer | `adamw_8bit` |
| Batch size | 4 |
| Gradient accumulation | 4 |
| 有效 batch size | 16 |
| Warmup | 0 |

不同规模模型会调整单卡 batch size，但数据格式、response-only loss、LoRA 目标模块和
一轮 SFT 思路基本一致。每个模型还使用不同随机种子，为最终集成提供差异性。

### 6. SFT 后如何得到分类分数

推理阶段并不要求模型真正生成一整段回答。推理代码取 assistant 第一个输出位置的
vocabulary logits，收集多个肯定和否定写法的首 token：

```text
肯定：Yes, YES, Y, yes, True
否定：No, NO, N, no, False
```

然后只在这些候选 token 上计算归一化分数，得到 `p_yes`。最后再在每条规则内部进行
rank normalization，生成提交分数。

因此生成式训练链路可以概括为：

```text
二元标签
  → Yes/No completion
  → chat template
  → 4-bit LoRA SFT
  → assistant 首 token logits
  → Yes/No 相对概率
  → 每条规则内排名
  → submission
```

需要特别区分：以上是 Qwen/Llama 的生成式 SFT。当前推荐 Colab 使用的 Ettin 是
encoder sequence classification，采用 BCE loss，并不执行聊天 SFT。两者共享相同的
数据构造和按规则排名逻辑，但模型训练目标不同。

## Ettin 单模型训练方法

Ettin 训练由单模型 runner 调用，统一入口位于
[`scripts/run_verified_ettin.py`](scripts/run_verified_ettin.py)。

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
- `run_manifest.json`：GitHub commit、Hugging Face 模型版本和数据规模；
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
jigsaw_training_method.ipynb      独立 Jupyter 主方法文档与可执行入口
scripts/build_project_method_notebook.py
scripts/run_verified_ettin.py     单模型闭环 runner
notebooks/verified_ettin_colab.ipynb
src/first_place/data.py           统一数据清洗和构造
src/first_place/ensemble.py       可选的多模型预测融合
src/first_place/validation.py     submission 格式与数值校验
tests/                            数据、提交和融合回归测试
```

## 项目范围

- 单模型 Colab 用于验证完整工程和训练思路，不代表多模型集成的最终效果。
- 当前仓库不声明未经同一评估环境验证的排行榜成绩。
- 只有在相同数据、指标和竞赛环境中提交后，才能进行可靠分数比较。
- 模型、依赖、数据统计和 GitHub commit 会写入运行清单，便于追踪每次实验。

