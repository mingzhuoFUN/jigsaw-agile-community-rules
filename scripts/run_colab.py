"""Run the exact active model sequence from the winning notebook on one Colab GPU."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

from huggingface_hub import snapshot_download


FAITHFUL_SEQUENCE = [
    ("train_qwen3_14b.py", "submission1.csv", "unsloth/Qwen3-14B-unsloth-bnb-4bit"),
    ("train_qwen2.5_14b.py", "submission2.csv", "unsloth/Qwen2.5-14B-Instruct-bnb-4bit"),
    ("train_qwen3_8b.py", "submission3.csv", "unsloth/Qwen3-8B-unsloth-bnb-4bit"),
    ("train_llama3_8b.py", "submission4.csv", "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit"),
    ("train_qwen3_4b.py", "submission5.csv", "unsloth/Qwen3-4B-Instruct-2507-unsloth-bnb-4bit"),
    ("train_ettin_400m.py", "submission7.csv", "jhu-clsp/ettin-encoder-400m"),
]


def prefetch_model(repo_id: str, attempts: int = 4) -> None:
    """Download to the shared HF cache with resumable retries."""
    for attempt in range(1, attempts + 1):
        try:
            print(f"Prefetching {repo_id} (attempt {attempt}/{attempts})", flush=True)
            path = snapshot_download(repo_id=repo_id, max_workers=4)
            print(f"Cached {repo_id} at {path}", flush=True)
            return
        except Exception as exc:
            if attempt == attempts:
                raise
            delay = 15 * attempt
            print(f"Download interrupted: {exc}\nRetrying in {delay}s...", flush=True)
            time.sleep(delay)


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
        "TOKENIZERS_PARALLELISM": "false",
    })
    # The reference notebook forced Kaggle's offline mode. Colab must download
    # public Hugging Face models, so remove these variables entirely. Some
    # downstream libraries treat even the string "0" as truthy.
    for variable in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE"):
        env.pop(variable, None)

    running = args.start_at is None
    for script_name, submission_name, model_id in FAITHFUL_SEQUENCE:
        if script_name == args.start_at:
            running = True
        if not running:
            continue
        prefetch_model(model_id)
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
            lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
            print("\n".join(lines[-120:]), file=sys.stderr, flush=True)
            raise SystemExit(f"{script_name} failed ({result.returncode}); see {log_path}")
        if not (args.output_dir / submission_name).exists():
            raise FileNotFoundError(f"{script_name} did not produce {submission_name}")
        if script_name == args.stop_after:
            break


if __name__ == "__main__":
    main()
