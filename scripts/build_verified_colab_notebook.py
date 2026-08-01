"""Generate the small, end-to-end verified Colab training notebook."""

from pathlib import Path

import nbformat as nbf


nb = nbf.v4.new_notebook()
nb["metadata"]["colab"] = {"name": "Verified Jigsaw Ettin training", "provenance": []}
nb["metadata"]["kernelspec"] = {"display_name": "Python 3", "name": "python3"}
nb["cells"] = [
    nbf.v4.new_markdown_cell(
        """# Verified GitHub → Hugging Face → Colab training path

This notebook proves the complete engineering chain with the first-place solution's
Ettin-400M component:

1. clone public code from GitHub;
2. install pinned dependencies without replacing Colab's CUDA Torch;
3. authenticate with Colab Secrets;
4. download Kaggle data;
5. download a pinned Hugging Face model revision;
6. run a real 32-row smoke training;
7. run full training and inference;
8. persist the model, logs, manifest, and submission to Google Drive.

Required Secret: `KAGGLE_API_TOKEN`. Recommended Secret: read-only `HF_TOKEN`."""
    ),
    nbf.v4.new_code_cell(
        """REPOSITORY = "mingzhuoFUN/jigsaw-agile-community-rules"
BRANCH = "main"
WORKDIR = "/content/jigsaw-verified"
DRIVE_OUTPUT_DIR = "/content/drive/MyDrive/jigsaw-verified-ettin"
RUN_FULL_TRAINING = True"""
    ),
    nbf.v4.new_code_cell(
        """import os, shutil, subprocess

clone_url = f"https://github.com/{REPOSITORY}.git"
if os.path.isdir(f"{WORKDIR}/.git"):
    subprocess.run(["git", "-C", WORKDIR, "pull", "--ff-only"], check=True)
else:
    shutil.rmtree(WORKDIR, ignore_errors=True)
    subprocess.run(["git", "clone", "--depth", "1", "--branch", BRANCH, clone_url, WORKDIR], check=True)
os.chdir(WORKDIR)
print("Commit:", subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip())"""
    ),
    nbf.v4.new_code_cell(
        """import torch
print("torch:", torch.__version__, "cuda:", torch.cuda.is_available())
if not torch.cuda.is_available() or "+cpu" in torch.__version__:
    raise RuntimeError("Select a GPU runtime, delete the current runtime, and start again.")
print("GPU:", torch.cuda.get_device_name(0))"""
    ),
    nbf.v4.new_code_cell(
        """%pip install -q -r requirements-verified-colab.txt
%pip install -q -e . --no-deps"""
    ),
    nbf.v4.new_code_cell(
        """import os, subprocess, sys
from pathlib import Path

src_path = str(Path(WORKDIR) / "src")
os.environ["PYTHONPATH"] = os.pathsep.join(
    [src_path, os.environ.get("PYTHONPATH", "")]
).rstrip(os.pathsep)
if src_path not in sys.path:
    sys.path.insert(0, src_path)
subprocess.run([sys.executable, "-m", "compileall", "-q", "src", "scripts"], check=True)
subprocess.run([sys.executable, "-m", "pytest", "-q"], check=True)
print("Repository import and tests passed.")"""
    ),
    nbf.v4.new_code_cell(
        """import os
from google.colab import userdata

kaggle_token = userdata.get("KAGGLE_API_TOKEN")
if not kaggle_token:
    raise ValueError("Add KAGGLE_API_TOKEN to Colab Secrets.")
os.environ["KAGGLE_API_TOKEN"] = kaggle_token

try:
    hf_token = userdata.get("HF_TOKEN")
except Exception:
    hf_token = None
if hf_token:
    os.environ["HF_TOKEN"] = hf_token
    print("Hugging Face authenticated.")
else:
    print("HF_TOKEN is absent; public anonymous download will be used.")"""
    ),
    nbf.v4.new_code_cell(
        """from pathlib import Path
import subprocess, zipfile

data_dir = Path(WORKDIR) / "data" / "raw"
data_dir.mkdir(parents=True, exist_ok=True)
archive = data_dir / "jigsaw-agile-community-rules.zip"
if not (data_dir / "train.csv").exists():
    subprocess.run([
        "kaggle", "competitions", "download",
        "-c", "jigsaw-agile-community-rules", "-p", str(data_dir)
    ], check=True)
    with zipfile.ZipFile(archive) as zf:
        zf.extractall(data_dir)
required = {"train.csv", "test.csv", "sample_submission.csv"}
assert required <= {path.name for path in data_dir.glob("*.csv")}
print("Competition data ready:", data_dir)"""
    ),
    nbf.v4.new_code_cell(
        """from google.colab import drive
from pathlib import Path

drive.mount("/content/drive")
output_dir = Path(DRIVE_OUTPUT_DIR)
output_dir.mkdir(parents=True, exist_ok=True)
print("Persistent output:", output_dir)"""
    ),
    nbf.v4.new_code_cell(
        """# Real low-cost training gate; downloads the same pinned HF model used below.
smoke_dir = Path("/content/jigsaw-ettin-smoke")
subprocess.run([
    sys.executable, "scripts/run_verified_ettin.py",
    "--data-dir", str(data_dir),
    "--output-dir", str(smoke_dir),
    "--smoke-rows", "32",
], check=True)
assert (smoke_dir / "submission7.csv").exists()
print("Smoke training passed.")"""
    ),
    nbf.v4.new_code_cell(
        """if RUN_FULL_TRAINING:
    subprocess.run([
        sys.executable, "scripts/run_verified_ettin.py",
        "--data-dir", str(data_dir),
        "--output-dir", str(output_dir),
    ], check=True)
    required = [
        output_dir / "submission7.csv",
        output_dir / "training.log",
        output_dir / "run_manifest.json",
        output_dir / "model" / "config.json",
    ]
    for path in required:
        assert path.exists() and path.stat().st_size > 0, path
    print("Full remote training complete.")
    for path in required:
        print(path, path.stat().st_size)
else:
    print("Smoke passed. Set RUN_FULL_TRAINING=True to execute full training.")"""
    ),
]

target = Path("notebooks/verified_ettin_colab.ipynb")
target.parent.mkdir(parents=True, exist_ok=True)
nbf.write(nb, target)
print(target)
