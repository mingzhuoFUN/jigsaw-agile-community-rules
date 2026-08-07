"""GitHub -> Hugging Face -> Colab -> Google Drive 单模型训练入口。"""

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

from first_place.data import build_training_frame
from first_place.validation import validate_submission


MODEL_ID = "jhu-clsp/ettin-encoder-400m"
# 固定模型提交版本，防止 Hugging Face 仓库更新后结果发生漂移。
MODEL_REVISION = "7662476d60abb071a5bd319c9f3074f3072c062d"


def download_model(attempts: int = 4) -> Path:
    from huggingface_hub import snapshot_download

    # 大模型下载可能因 Colab 网络抖动中断，采用递增等待时间自动重试。
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


def run_training(data_dir: Path, output_dir: Path, smoke_rows: int) -> None:
    root = Path(__file__).resolve().parents[1]
    script = root / "scripts" / "winner" / "train_ettin_400m.py"
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = download_model()
    # 通过环境变量向训练子进程传递路径和 smoke 配置，避免修改训练脚本常量。
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
    # Colab 需要在线下载模型；即使离线变量值为 "0"，部分依赖仍可能把它当成开启。
    for variable in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE"):
        env.pop(variable, None)

    command = [sys.executable, "-u", str(script), "--train_then_infer"]
    if smoke_rows:
        # smoke 模式只验证训练和推理闭环，不保存体积较大的模型文件。
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
        # 同时输出到 Colab 控制台和持久化日志，便于观察进度与排查失败。
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
            log.flush()
        return_code = process.wait()
    if return_code:
        raise SystemExit(f"Training failed with exit code {return_code}; see {log_path}")

    # 训练成功后立即校验 submission，防止把错误产物写入 Google Drive。
    submission_path = output_dir / "submission7.csv"
    submission = validate_submission(submission_path, data_dir)
    training_frame = build_training_frame(data_dir, example_repeats=2, seed=3001)
    # manifest 记录模型、数据和运行方式，为后续实验对比提供依据。
    manifest = {
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "smoke_rows": smoke_rows,
        "train_rows": int(len(pd.read_csv(data_dir / "train.csv"))),
        "constructed_training_rows": int(len(training_frame)),
        "constructed_source_counts": {
            key: int(value)
            for key, value in training_frame["source"].value_counts().items()
        },
        "test_example_repeats": 2,
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
