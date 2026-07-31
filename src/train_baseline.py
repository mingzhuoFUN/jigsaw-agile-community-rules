from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.pipeline import FeatureUnion

from features import build_pair_text


def make_model() -> Pipeline:
    return Pipeline(
        steps=[
            (
                "features",
                FeatureUnion(
                    [
                        (
                            "word",
                            TfidfVectorizer(
                                analyzer="word",
                                ngram_range=(1, 2),
                                min_df=2,
                                max_df=0.95,
                                sublinear_tf=True,
                                strip_accents="unicode",
                                max_features=80_000,
                            ),
                        ),
                        (
                            "char",
                            TfidfVectorizer(
                                analyzer="char_wb",
                                ngram_range=(3, 5),
                                min_df=2,
                                sublinear_tf=True,
                                strip_accents="unicode",
                                max_features=80_000,
                            ),
                        ),
                    ]
                ),
            ),
            (
                "clf",
                CalibratedClassifierCV(
                    estimator=LogisticRegression(
                        C=2.0,
                        max_iter=2_000,
                        class_weight="balanced",
                        solver="liblinear",
                        random_state=42,
                    ),
                    method="sigmoid",
                    cv=3,
                ),
            ),
        ]
    )


def train_and_predict(data_dir: Path, output_dir: Path, folds: int) -> dict[str, float]:
    train = pd.read_csv(data_dir / "train.csv")
    test = pd.read_csv(data_dir / "test.csv")
    sample_submission = pd.read_csv(data_dir / "sample_submission.csv")

    x_train = build_pair_text(train)
    y = train["rule_violation"].astype(int)
    x_test = build_pair_text(test)

    model = make_model()
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)
    oof = cross_val_predict(model, x_train, y, cv=cv, method="predict_proba")[:, 1]
    auc = roc_auc_score(y, oof)

    model.fit(x_train, y)
    test_pred = model.predict_proba(x_test)[:, 1]

    output_dir.mkdir(parents=True, exist_ok=True)
    submission = sample_submission.copy()
    submission["rule_violation"] = test_pred
    submission.to_csv(output_dir / "submission_baseline.csv", index=False)

    pd.DataFrame({"row_id": train["row_id"], "rule_violation": y, "oof_pred": oof}).to_csv(
        output_dir / "oof_baseline.csv", index=False
    )

    metrics = {"cv_auc": float(auc), "folds": folds, "train_rows": int(len(train)), "test_rows": int(len(test))}
    (output_dir / "metrics_baseline.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--folds", type=int, default=5)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    metrics = train_and_predict(args.data_dir, args.output_dir, args.folds)
    print(json.dumps(metrics, indent=2))
