# Jigsaw Agile Community Rules 复现笔记

Kaggle: https://www.kaggle.com/competitions/jigsaw-agile-community-rules

[在 Google Colab 中打开](https://colab.research.google.com/github/mingzhuoFUN/jigsaw-agile-community-rules/blob/main/notebooks/first_place_reproduction_colab.ipynb)

## 任务

给定 Reddit 评论 `body`、社区 `subreddit`、待判断的社区规则 `rule`，以及该规则下的正负示例，预测这条评论是否违反该规则。目标列是 `rule_violation`，提交文件需要输出每个 `row_id` 的违规概率。

竞赛评估指标是 AUC。页面说明里特别强调：训练集只包含两条规则，但测试集会包含训练中没有见过的其他规则，所以模型必须利用规则文本和示例泛化，而不是把规则当成固定类别记住。

## 当前数据

- `train.csv`: 2029 行，9 列
- `test.csv`: 10 行，8 列
- `sample_submission.csv`: 10 行，2 列
- 训练标签较均衡：`1` 为 1031 行，`0` 为 998 行
- 训练集中两条规则：
  - No legal advice
  - No Advertising

## 环境

```powershell
pip install -r requirements.txt
```

## 下载数据

需要先在 Kaggle 登录并接受竞赛规则，然后运行：

```powershell
.\download_data.ps1
```

也可以手动运行：

```powershell
kaggle competitions download -c jigsaw-agile-community-rules -p data/raw
Expand-Archive -LiteralPath data/raw/jigsaw-agile-community-rules.zip -DestinationPath data/raw -Force
```

## 查看数据

```powershell
$env:PYTHONIOENCODING='utf-8'
python src/eda.py
```

## Baseline

第一版 baseline 使用：

- 文本构造：`COMMENT + SUBREDDIT + RULE + VIOLATING EXAMPLES + NON-VIOLATING EXAMPLES`
- 特征：TF-IDF word 1-2 gram + char_wb 3-5 gram
- 模型：Logistic Regression + sigmoid calibration
- 验证：Stratified K-Fold AUC

```powershell
python src/train_baseline.py
```

输出：

- `outputs/submission_baseline.csv`
- `outputs/oof_baseline.csv`
- `outputs/metrics_baseline.json`

提交：

```powershell
kaggle competitions submit -c jigsaw-agile-community-rules -f outputs/submission_baseline.csv -m "tfidf logistic baseline"
```

## 后续复现路线

1. 强化验证：按 `rule` 或 `subreddit` 做留出，模拟未见规则/社区。
2. 特征改进：分别编码 comment、rule、positive examples、negative examples，加入相似度特征。
3. Transformer baseline：使用 DeBERTa/MiniLM 做 pair classification。
4. Few-shot/LLM 路线：把正负例作为上下文，做零样本或小样本推理，再校准概率。
5. 集成：TF-IDF、Transformer、规则关键词、相似度模型做 rank averaging。

## 第一名方案完整复现

参考 notebook：`1st-place-code (1).ipynb`。它不是单模型方案，而是 6 个生成式
LoRA 模型与 1 个 Ettin-400M 编码器的集成。训练时把测试集提供的正反例转成带标签
样本，并重复 3 次以增强未见规则上的泛化。

项目化复现代码位于：

- `notebooks/first_place_reproduction_colab.ipynb`：Colab 完整训练入口。
- `REPRODUCTION.md`：参考版本、实际启用模型、参数和已知歧义。
- `reference/notebook_cells/`：原 notebook 中全部 21 个 `%%writefile` 脚本的冻结副本。
- `scripts/winner/`：仅修改路径和在线模型来源的可运行脚本。
- `scripts/run_colab.py`：在单 GPU Colab 上顺序执行原方案实际启用的 6 个模型。
- `configs/first_place.json`：原方案的模型矩阵与关键参数。
- `src/first_place/data.py`：提示词和训练样本构造。
- `src/first_place/ensemble.py`：按 `row_id` 对齐并加权融合各模型预测。
- `tests/test_first_place.py`：数据增强和融合逻辑的回归测试。

Colab 使用步骤：

1. 打开上面的 Colab 链接并选择 GPU 运行时，14B 模型建议 A100/高内存实例。
2. 在 Colab Secrets 添加 `KAGGLE_API_TOKEN`，并确保 Kaggle 账号已接受竞赛规则。
3. 依次运行单元格；smoke test 通过后将 `RUN_FULL_TRAINING` 改为 `True`。
4. 日志、各模型预测与最终 `submission.csv` 会写入 Google Drive。

Colab 单 GPU 入口按顺序执行模型，只改变原双 GPU notebook 的并发调度，不修改模型、
训练数据、随机种子或超参数。原 notebook 实际启用 6 个预测模型；Phi-4 和
Llama-3.2-3B 虽有定义但在编排中被注释，详情见 `REPRODUCTION.md`。

融合已有预测：

```powershell
python -m first_place.ensemble `
  --input-dir outputs/first_place `
  --output outputs/submission_first_place.csv
```

notebook 原始融合权重之和为 `1.1`，项目代码会自动归一化，同时允许只融合当前已经
完成的模型，避免输出概率超出预期尺度。
