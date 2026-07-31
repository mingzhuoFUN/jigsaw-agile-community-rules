# Jigsaw Agile Community Rules 复现笔记

Kaggle: https://www.kaggle.com/competitions/jigsaw-agile-community-rules

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
