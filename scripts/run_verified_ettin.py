"""Verified single-model training path for GitHub -> HF -> Colab -> Drive."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


MODEL_ID = "jhu-clsp/ettin-encoder-400m"
MODEL_REVISION = "7662476d60abb071a5bd319c9f3074f3072c062d"


def download_model(attempts: int = 4) -> Path:
    from huggingface_hub import snapshot_download

    for attempt in range(1, attempts + 1):
        try:
            print(f"Downloading {MODEL_ID}@{MODEL_REVISION[:8]} ({attempt}/{attempts})", flush=True)
            return Path(
                snapshot_download(
                    repo_id=MODEL_ID,
                    revision=MODEL_REVISION,
                    max_workers=4,
                )
            )
        except Exception as exc:
            if attempt == attempts:
                raise
            delay = 15 * attempt
            print(f"Download interrupted: {exc}\nRetrying in {delay}s...", flush=True)
            time.sleep(delay)
    raise RuntimeError("unreachable")


def validate_submission(path: Path, data_dir: Path) -> pd.DataFrame:
    submission = pd.read_csv(path)
    sample = pd.read_csv(data_dir / "sample_submission.csv")
    if list(submission.columns) != ["row_id", "rule_violation"]:
        raise ValueError(f"Unexpected submission columns: {submission.columns.tolist()}")
    if len(submission) != len(sample):
        raise ValueError(f"Expected {len(sample)} predictions, got {len(submission)}")
    if set(submission["row_id"]) != set(sample["row_id"]):
        raise ValueError("Submission row_id values do not match sample_submission.csv")
    if submission["row_id"].duplicated().any():
        raise ValueError("Duplicate row_id values in submission")
    if not submission["rule_violation"].notna().all():
        raise ValueError("Submission contains missing predictions")
    if not submission["rule_violation"].between(0, 1).all():
        raise ValueError("Predictions must be in [0, 1]")
    return submission


def run_training(data_dir: Path, output_dir: Path, smoke_rows: int) -> None:
    root = Path(__file__).resolve().parents[1]
    script = root / "scripts" / "winner" / "train_ettin_400m.py"
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = download_model()
    env = os.environ.copy()
    env.update(
        {
            "BASE_MODEL_PATH": str(model_path),
            "DATA_PATH": str(data_dir.resolve()),
            "OUTPUT_DIR": str((output_dir / "model").resolve()),
            "SMOKE_ROWS": str(smoke_rows),
            "TOKENIZERS_PARALLELISM": "false",
        }
    )
    for variable in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE"):
        env.pop(variable, None)

    command = [sys.executable, "-u", str(script), "--train_then_infer"]
    if smoke_rows:
        command.append("--no_save")

    log_path = output_dir / "training.log"
    print(f"Live log: {log_path}", flush=True)
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            command,
            cwd=output_dir,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
            log.flush()
        return_code = process.wait()
    if return_code:
        raise SystemExit(f"Training failed with exit code {return_code}; see {log_path}")

    submission_path = output_dir / "submission7.csv"
    submission = validate_submission(submission_path, data_dir)
    manifest = {
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "smoke_rows": smoke_rows,
        "train_rows": int(len(pd.read_csv(data_dir / "train.csv"))),
        "test_rows": int(len(submission)),
        "submission": str(submission_path),
        "model_saved": not bool(smoke_rows),
    }
    (output_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--smoke-rows", type=int, default=0)
    args = parser.parse_args()
    run_training(args.data_dir, args.output_dir, args.smoke_rows)


if __name__ == "__main__":
    main()
