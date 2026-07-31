"""Generate the deterministic Colab entrypoint."""

from pathlib import Path

import nbformat as nbf


nb = nbf.v4.new_notebook()
nb["metadata"]["colab"] = {"name": "First-place Jigsaw reproduction", "provenance": []}
nb["metadata"]["kernelspec"] = {"display_name": "Python 3", "name": "python3"}
nb["cells"] = [
    nbf.v4.new_markdown_cell(
        """# Jigsaw first-place reproduction

Faithful Colab entrypoint for the supplied winning notebook. Before running:

1. Select a GPU runtime.
2. Add `GITHUB_TOKEN` to Colab Secrets because the repository is private.
3. Add `KAGGLE_API_TOKEN` to Colab Secrets and accept the competition rules.
4. Set `RUN_FULL_TRAINING = True` only after the smoke checks pass.

Large models may require an A100/high-memory runtime. The runner is sequential, so it also
works on a single-GPU Colab runtime without changing model semantics."""
    ),
    nbf.v4.new_code_cell(
        """REPOSITORY = "mingzhuoFUN/jigsaw-agile-community-rules"
BRANCH = "main"
WORKDIR = "/content/jigsaw-first-place"
DRIVE_ARTIFACT_DIR = "/content/drive/MyDrive/jigsaw-first-place"
RUN_FULL_TRAINING = False"""
    ),
    nbf.v4.new_code_cell(
        """from google.colab import userdata
from urllib.parse import quote
import os, shutil, subprocess

github_token = userdata.get("GITHUB_TOKEN")
if not github_token:
    raise ValueError("Add GITHUB_TOKEN in Colab Secrets; the repository is private.")
clone_url = f"https://x-access-token:{quote(github_token, safe='')}@github.com/{REPOSITORY}.git"
if os.path.isdir(f"{WORKDIR}/.git"):
    subprocess.run(["git", "-C", WORKDIR, "pull", "--ff-only"], check=True)
else:
    shutil.rmtree(WORKDIR, ignore_errors=True)
    subprocess.run(["git", "clone", "--depth", "1", "--branch", BRANCH, clone_url, WORKDIR], check=True)
os.chdir(WORKDIR)"""
    ),
    nbf.v4.new_code_cell(
        """%pip install -q -U pip
%pip install -q -r requirements-colab.txt
%pip install -q -e .
%pip install -q unsloth"""
    ),
    nbf.v4.new_code_cell(
        """import subprocess, sys, torch
subprocess.run([sys.executable, "-m", "compileall", "-q", "src", "scripts"], check=True)
subprocess.run([sys.executable, "-m", "pytest", "-q"], check=True)
import first_place
print("torch:", torch.__version__, "cuda:", torch.cuda.is_available())
if not torch.cuda.is_available():
    raise RuntimeError("Select a GPU runtime before training.")"""
    ),
    nbf.v4.new_code_cell(
        """from google.colab import userdata
from pathlib import Path
import os, subprocess, zipfile

kaggle_token = userdata.get("KAGGLE_API_TOKEN")
if not kaggle_token:
    raise ValueError("Add KAGGLE_API_TOKEN in Colab Secrets.")
os.environ["KAGGLE_API_TOKEN"] = kaggle_token
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
print(sorted(p.name for p in data_dir.glob("*.csv")))"""
    ),
    nbf.v4.new_code_cell(
        """from google.colab import drive
from pathlib import Path
drive.mount("/content/drive")
artifact_dir = Path(DRIVE_ARTIFACT_DIR)
artifact_dir.mkdir(parents=True, exist_ok=True)
print("Persistent artifacts:", artifact_dir)"""
    ),
    nbf.v4.new_code_cell(
        """from pathlib import Path
from first_place.data import build_training_frame
frame = build_training_frame(Path(WORKDIR) / "data" / "raw")
assert {"body", "rule", "rule_violation", "source"} <= set(frame.columns)
assert frame["rule_violation"].isin([0, 1]).all()
print(frame.shape, frame["source"].value_counts().to_dict())"""
    ),
    nbf.v4.new_code_cell(
        """# Low-cost real training gate: 32 balanced rows through the Ettin training/inference path.
smoke_dir = Path("/content/jigsaw-smoke")
smoke_dir.mkdir(exist_ok=True)
smoke_env = os.environ.copy()
smoke_env.update({
    "DATA_PATH": str(Path(WORKDIR) / "data" / "raw"),
    "OUTPUT_DIR": str(smoke_dir / "model"),
    "SMOKE_ROWS": "32",
})
subprocess.run(
    [sys.executable, str(Path(WORKDIR) / "scripts/winner/train_ettin_400m.py")],
    cwd=smoke_dir, env=smoke_env, check=True,
)
smoke_submission = __import__("pandas").read_csv(smoke_dir / "submission7.csv")
assert len(smoke_submission) > 0
assert smoke_submission["rule_violation"].notna().all()
print("Real training smoke test passed:", smoke_submission.shape)"""
    ),
    nbf.v4.new_code_cell(
        """if RUN_FULL_TRAINING:
    subprocess.run([
        sys.executable, "scripts/run_colab.py",
        "--data-dir", str(Path(WORKDIR) / "data" / "raw"),
        "--output-dir", str(artifact_dir),
    ], check=True)
else:
    print("Smoke checks passed. Set RUN_FULL_TRAINING=True and rerun this cell.")"""
    ),
    nbf.v4.new_code_cell(
        """if RUN_FULL_TRAINING:
    subprocess.run([
        sys.executable, "-m", "first_place.ensemble",
        "--input-dir", str(artifact_dir),
        "--output", str(artifact_dir / "submission.csv"),
    ], check=True)
    submission = __import__("pandas").read_csv(artifact_dir / "submission.csv")
    assert submission["rule_violation"].notna().all()
    assert submission["row_id"].is_unique
    print(submission.head(), artifact_dir / "submission.csv")"""
    ),
]

target = Path("notebooks/first_place_reproduction_colab.ipynb")
target.parent.mkdir(parents=True, exist_ok=True)
nbf.write(nb, target)
print(target)
