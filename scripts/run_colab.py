"""Run the exact active model sequence from the winning notebook on one Colab GPU."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


FAITHFUL_SEQUENCE = [
    ("train_qwen3_14b.py", "submission1.csv"),
    ("train_qwen2.5_14b.py", "submission2.csv"),
    ("train_qwen3_8b.py", "submission3.csv"),
    ("train_llama3_8b.py", "submission4.csv"),
    ("train_qwen3_4b.py", "submission5.csv"),
    ("train_ettin_400m.py", "submission7.csv"),
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--start-at", choices=[item[0] for item in FAITHFUL_SEQUENCE])
    parser.add_argument("--stop-after", choices=[item[0] for item in FAITHFUL_SEQUENCE])
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    scripts = root / "scripts" / "winner"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update({
        "DATA_PATH": str(args.data_dir.resolve()),
        "HF_HUB_OFFLINE": "0",
        "TRANSFORMERS_OFFLINE": "0",
        "HF_DATASETS_OFFLINE": "0",
        "TOKENIZERS_PARALLELISM": "false",
    })

    running = args.start_at is None
    for script_name, submission_name in FAITHFUL_SEQUENCE:
        if script_name == args.start_at:
            running = True
        if not running:
            continue
        log_path = args.output_dir / f"{Path(script_name).stem}.log"
        print(f"\n=== {script_name} ===", flush=True)
        with log_path.open("w", encoding="utf-8") as log:
            result = subprocess.run(
                [sys.executable, "-u", str(scripts / script_name)],
                cwd=args.output_dir,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=False,
            )
        if result.returncode:
            raise SystemExit(f"{script_name} failed ({result.returncode}); see {log_path}")
        if not (args.output_dir / submission_name).exists():
            raise FileNotFoundError(f"{script_name} did not produce {submission_name}")
        if script_name == args.stop_after:
            break


if __name__ == "__main__":
    main()
