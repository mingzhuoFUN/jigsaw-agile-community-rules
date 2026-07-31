import os, math, time, argparse

# ----------------------
# Constants
# ----------------------
BASE_MODEL_PATH = os.environ.get("BASE_MODEL_PATH", "jhu-clsp/ettin-encoder-400m")
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "output_deberta/")
DATA_PATH = os.environ.get("DATA_PATH", "data/raw")

POSITIVE_ANSWER = "Yes"
NEGATIVE_ANSWER = "No"
COMPLETE_PHRASE = "Answer:"
BASE_PROMPT = "Reddit moderation: Does the comment violate the rule? Answer 'Yes' or 'No' only."

# ----------------------
# Imports
# ----------------------
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ.setdefault("NCCL_IB_DISABLE", "1")
os.environ.setdefault("NCCL_ASYNC_ERROR_HANDLING", "1")

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoTokenizer, AutoConfig, AutoModelForSequenceClassification,
    get_cosine_schedule_with_warmup
)

# Speed-friendly defaults
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.benchmark = True
if hasattr(torch, "set_float32_matmul_precision"):
    torch.set_float32_matmul_precision("high")

# -------- Config knobs (env) ----------
MAX_LENGTH = int(os.environ.get("MAX_LENGTH", "512"))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "8"))     # train batch size
EPOCHS     = int(os.environ.get("EPOCHS", "1"))
LR         = float(os.environ.get("LR", "2e-5"))
WD         = float(os.environ.get("WD", "0.01"))
WARMUP     = float(os.environ.get("WARMUP_RATIO", "0.00"))  # fraction of total steps
GRAD_ACCUM = int(os.environ.get("GRAD_ACCUM", "2"))
GRADIENT_CHECKPOINTING = os.environ.get("GRADIENT_CHECKPOINTING", "0") == "1"

# Inference knobs
INFER_MAX_LENGTH = int(os.environ.get("INFER_MAX_LENGTH", str(MAX_LENGTH)))
INFER_BATCH_SIZE = int(os.environ.get("INFER_BATCH_SIZE", "32"))

# ----------------------
# Utils
# ----------------------
def build_prompt(row):
    # Kept for parity; not used by this pipeline
    return f"""
{BASE_PROMPT}

Comment: {row["body"]}

rule: {row["rule"]}
---
{COMPLETE_PHRASE}"""

def get_dataframe_to_train(data_path):
    train_dataset = pd.read_csv(f"{data_path}/train.csv")
    test_dataset = pd.read_csv(f"{data_path}/test.csv")

    flatten = []

    # base train rows
    base = train_dataset[["body", "rule", "rule_violation"]].copy()
    base["source"] = "train"
    flatten.append(base)

    # upsample target block (test examples) by labeling them now
    for violation_type in ["positive", "negative"]:
        for i in range(1, 3):
            col = f"{violation_type}_example_{i}"
            sub_dataset = test_dataset[[col, "rule"]].copy()
            sub_dataset = sub_dataset.rename(columns={col: "body"})
            sub_dataset["rule_violation"] = 1 if violation_type == "positive" else 0
            sub_dataset["source"] = "test_examples"
            flatten.append(sub_dataset)

    dataframe = pd.concat(flatten, axis=0, ignore_index=True)
    dataframe = dataframe.drop_duplicates(ignore_index=True)

    # upsample test_examples once more (2x extra copies -> appears 3x total)
    test_rows = dataframe[dataframe["source"] == "test_examples"]
    if not test_rows.empty:
        dataframe = pd.concat([dataframe, test_rows], axis=0, ignore_index=True)

    dataframe = dataframe.sample(frac=1.0, random_state=3001).reset_index(drop=True)
    dataframe = dataframe.drop(columns=["source"])
    return dataframe

def build_classification_dataframe(df, tok_sep_token="</s>"):
    """
    Build a dataframe for classification using text = rule + [SEP] + comment.
    """
    df = df.copy()
    sep = tok_sep_token if tok_sep_token else "</s>"
    df["text"] = df["rule"].astype(str) + sep + df["body"].astype(str)
    cols = ["text"]
    if "rule_violation" in df:
        cols.append("rule_violation")
    return df[cols]

# ----------------------
# Dataset & Collate
# ----------------------
class TxtClsDataset(Dataset):
    def __init__(self, texts, labels=None):
        self.texts = list(texts)
        self.labels = None if labels is None else list(labels)

    def __len__(self): return len(self.texts)

    def __getitem__(self, i):
        if self.labels is None:
            return {"text": self.texts[i]}
        return {"text": self.texts[i], "label": float(self.labels[i])}

def collate_fn_builder(tokenizer, max_length):
    def _fn(batch):
        texts = [b["text"] for b in batch]
        enc = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        out = {"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"]}
        if "label" in batch[0]:
            labels = torch.tensor([b["label"] for b in batch], dtype=torch.float32).unsqueeze(-1)
            out["labels"] = labels
        return out
    return _fn

# ----------------------
# Training (single-GPU) -> returns model & tokenizer so you can infer in-memory
# ----------------------
def train_single_gpu(no_save: bool = False):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Data
    raw_df = get_dataframe_to_train(DATA_PATH)
    smoke_rows = int(os.environ.get("SMOKE_ROWS", "0"))
    if smoke_rows:
        raw_df = raw_df.groupby("rule_violation", group_keys=False).apply(
            lambda group: group.sample(
                n=min(len(group), max(1, smoke_rows // 2)),
                random_state=3001,
            )
        ).reset_index(drop=True)

    # Tokenizer first, to know SEP token for concatenation
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH, use_fast=True, trust_remote_code=False)
    sep_token = tokenizer.sep_token if tokenizer.sep_token is not None else "</s>"

    df = build_classification_dataframe(raw_df, tok_sep_token=sep_token)

    train_ds = TxtClsDataset(df["text"], df["rule_violation"])

    collate_fn = collate_fn_builder(tokenizer, MAX_LENGTH)
    loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        collate_fn=collate_fn,
        drop_last=False,
    )

    # Model
    cfg = AutoConfig.from_pretrained(BASE_MODEL_PATH, num_labels=1, problem_type=None)  # we'll set loss manually
    model = AutoModelForSequenceClassification.from_pretrained(BASE_MODEL_PATH, config=cfg)
    if GRADIENT_CHECKPOINTING:
        model.gradient_checkpointing_enable()
    model.to(device)

    # Optimizer / Scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    total_steps = max(1, EPOCHS * len(loader) // max(1, GRAD_ACCUM))
    warmup_steps = int(WARMUP * total_steps)
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

    # Loss
    bce = torch.nn.BCEWithLogitsLoss()

    # AMP
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))
    model.train()

    global_step = 0
    t0 = time.time()
    optimizer.zero_grad(set_to_none=True)

    for epoch in range(EPOCHS):
        running = 0.0
        for step, batch in enumerate(loader):
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)  # shape (B,1)

            with torch.cuda.amp.autocast(dtype=torch.float16, enabled=(device.type == "cuda")):
                out = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = out.logits  # (B,1)
                loss = bce(logits, labels)

            loss = loss / max(1, GRAD_ACCUM)
            scaler.scale(loss).backward()

            if (step + 1) % GRAD_ACCUM == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                global_step += 1

            running += loss.item() * max(1, GRAD_ACCUM)

            if global_step > 0 and global_step % 10 == 0:
                lr_now = scheduler.get_last_lr()[0]
                print(f"[epoch {epoch+1}] step {global_step}/{total_steps} | lr={lr_now:.6e} | loss={running/10.0:.4f}")
                running = 0.0

        print(f"Epoch {epoch+1} done in {(time.time()-t0)/60:.1f} min")

    if not no_save:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        model.save_pretrained(OUTPUT_DIR)
        tokenizer.save_pretrained(OUTPUT_DIR)
        print(f"Saved model + tokenizer to {OUTPUT_DIR}")
    else:
        print("Skipping save (in-memory use only).")

    # Return in-memory artifacts for immediate inference
    return model, tokenizer

# ----------------------
# Inference (single-GPU)
# - If model/tokenizer are provided, uses them in-memory (no disk I/O).
# - Otherwise, loads from OUTPUT_DIR.
# ----------------------
def _bucket_by_length(tok, texts, max_length=512):
    lens = tok(texts, return_length=True, truncation=True, max_length=max_length)
    order = sorted(range(len(texts)), key=lambda k: lens["length"][k], reverse=True)
    return order

def infer_single_gpu(model: AutoModelForSequenceClassification = None,
                     tokenizer: AutoTokenizer = None):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if model is None or tokenizer is None:
        # fall back to disk
        tokenizer = AutoTokenizer.from_pretrained(OUTPUT_DIR, use_fast=True)
        cfg = AutoConfig.from_pretrained(OUTPUT_DIR)
        model = AutoModelForSequenceClassification.from_pretrained(OUTPUT_DIR, config=cfg)
        print(f"Loaded model + tokenizer from {OUTPUT_DIR}")

    model.eval()
    model.to(device)

    sep_token = tokenizer.sep_token if tokenizer.sep_token is not None else "</s>"

    # Load test
    df = pd.read_csv(f"{DATA_PATH}/test.csv")

    # --- robust column handling ---
    if "body" in df.columns:
        text_col = "body"
    elif "comment" in df.columns:
        text_col = "comment"
    elif "text" in df.columns:
        text_col = "text"
    else:
        raise KeyError(f"Could not find a text column. Available columns: {list(df.columns)}")

    if "rule" not in df.columns:
        raise KeyError("Expected 'rule' column in test.csv but did not find it.")

    # Build the classification dataframe using rule + [SEP] + body/comment
    tmp = df.rename(columns={text_col: "body"})  # utils expects 'body'
    cls_df = build_classification_dataframe(tmp, tok_sep_token=sep_token)

    texts = list(cls_df["text"])
    rows  = list(df["row_id"]) if "row_id" in df.columns else list(range(len(df)))
    rules = list(df["rule"])

    order = _bucket_by_length(tokenizer, texts, max_length=INFER_MAX_LENGTH)
    texts = [texts[i] for i in order]
    rows  = [rows[i]  for i in order]
    rules = [rules[i] for i in order]

    out_ids, out_probs, out_rules = [], [], []

    for i in range(0, len(texts), INFER_BATCH_SIZE):
        batch_texts = texts[i:i+INFER_BATCH_SIZE]
        enc = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=INFER_MAX_LENGTH,
            pad_to_multiple_of=8,
            return_tensors="pt",
        )
        enc = {k: v.to(device, non_blocking=True) for k, v in enc.items()}

        with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.float16, enabled=(device.type == "cuda")):
            logits = model(**enc).logits  # (B,1)
            probs = torch.sigmoid(logits).squeeze(-1)  # (B,)

        out_probs.extend(probs.float().cpu().tolist())
        out_ids.extend(rows[i:i+INFER_BATCH_SIZE])
        out_rules.extend(rules[i:i+INFER_BATCH_SIZE])

    out_df = pd.DataFrame({"row_id": out_ids, "rule": out_rules, "rule_violation": out_probs})

    # Per-rule ranks scaled to [0,1]
    r = out_df.groupby("rule")["rule_violation"].rank(method="average", ascending=True)
    n = out_df.groupby("rule")["rule_violation"].transform("size")
    denom = (n - 1).where(n > 1, 1)
    out_df["rule_violation"] = (r - 1) / denom

    out_df[["row_id", "rule_violation"]].sort_values("row_id").to_csv("submission7.csv", index=False)
    print("Wrote submission.csv")

# ----------------------
# CLI
# ----------------------
def parse_args():
    p = argparse.ArgumentParser(description="Single-GPU DeBERTa train + inference (in-memory capable)")
    p.add_argument("--do_train", action="store_true", help="Run training")
    p.add_argument("--do_infer", action="store_true", help="Run inference (loads from disk if no in-memory model)")
    p.add_argument("--train_then_infer", action="store_true",
                   help="Run training then inference (passes the trained model in-memory; no disk I/O unless you omit --no_save).")
    p.add_argument("--no_save", action="store_true", help="Do not save model/tokenizer after training.")
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    # Default behavior: train_then_infer with no-save (most efficient path)
    if not any([args.do_train, args.do_infer, args.train_then_infer]):
        args.train_then_infer = True
        args.no_save = True

    if args.do_train and not args.do_infer and not args.train_then_infer:
        train_single_gpu(no_save=args.no_save)

    elif args.do_infer and not args.do_train and not args.train_then_infer:
        # inference only -> will load from OUTPUT_DIR
        infer_single_gpu(model=None, tokenizer=None)

    elif args.train_then_infer:
        model, tok = train_single_gpu(no_save=args.no_save)
        # Pass in-memory model + tokenizer (no save/load needed)
        infer_single_gpu(model=model, tokenizer=tok)
