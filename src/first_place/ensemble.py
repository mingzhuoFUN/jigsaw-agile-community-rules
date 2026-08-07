from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


DEFAULT_WEIGHTS = {
    "submission1.csv": 0.20,
    "submission2.csv": 0.20,
    "submission3.csv": 0.20,
    "submission4.csv": 0.20,
    "submission5.csv": 0.10,
    "submission6.csv": 0.10,
    "submission7.csv": 0.10,
}


def blend(input_dir: Path, output_path: Path, weights: dict[str, float]) -> pd.DataFrame:
    # 允许只融合当前已经生成的预测文件，并重新归一化可用权重。
    available = {name: weight for name, weight in weights.items() if (input_dir / name).exists()}
    if not available:
        raise FileNotFoundError(f"No submission files found in {input_dir}")

    total = sum(available.values())
    normalized = {name: weight / total for name, weight in available.items()}
    result: pd.DataFrame | None = None
    for name, weight in normalized.items():
        current = pd.read_csv(input_dir / name)[["row_id", "rule_violation"]]
        current = current.rename(columns={"rule_violation": name})
        # 按 row_id 对齐而不是依赖 CSV 行顺序；one_to_one 会同时检查重复 row_id。
        result = current if result is None else result.merge(current, on="row_id", validate="one_to_one")

    assert result is not None
    result["rule_violation"] = sum(result[name] * weight for name, weight in normalized.items())
    submission = result[["row_id", "rule_violation"]].sort_values("row_id")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(output_path, index=False)
    return submission


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=Path("outputs/first_place"))
    parser.add_argument("--output", type=Path, default=Path("outputs/submission_first_place.csv"))
    args = parser.parse_args()
    result = blend(args.input_dir, args.output, DEFAULT_WEIGHTS)
    print(f"Wrote {len(result)} predictions to {args.output}")


if __name__ == "__main__":
    main()
