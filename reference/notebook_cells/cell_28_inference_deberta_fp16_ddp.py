import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch, torch.distributed as dist
import pandas as pd
from datetime import timedelta
from transformers import AutoTokenizer, AutoConfig, AutoModelForSequenceClassification

from utils import build_classification_dataframe
from constants import DATA_PATH, LORA_PATH, BASE_MODEL_PATH

torch.backends.cuda.matmul.allow_tf32 = True
if hasattr(torch, "set_float32_matmul_precision"):
    torch.set_float32_matmul_precision("high")

MAX_LENGTH = int(os.environ.get("MAX_LENGTH", "512"))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "32"))

def setup_ddp():
    if "RANK" in os.environ:
        world = int(os.environ["WORLD_SIZE"])
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl", timeout=timedelta(minutes=30))
    else:
        world, rank, local_rank = 1, 0, 0
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    return world, rank, local_rank, device

def bucket_by_length(tok, texts, max_length=512):
    lens = tok(texts, return_length=True, truncation=True, max_length=max_length)
    order = sorted(range(len(texts)), key=lambda k: lens["length"][k], reverse=True)
    return order

def main():
    world, rank, local_rank, device = setup_ddp()

    tokenizer = AutoTokenizer.from_pretrained(LORA_PATH, use_fast=True)
    sep_token = tokenizer.sep_token if tokenizer.sep_token is not None else "</s>"

    cfg = AutoConfig.from_pretrained(LORA_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(LORA_PATH, config=cfg)
    model.eval()
    model.to(device)

    # Load test
    df = pd.read_csv(f"{DATA_PATH}/test.csv")

    # --- robust column handling ---
    # choose the comment/body column
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

    # DDP shard
    idx = list(range(len(texts)))
    shard_idx   = idx[rank::world]
    shard_texts = [texts[i] for i in shard_idx]
    shard_rows  = [rows[i]  for i in shard_idx]
    shard_rules = [rules[i] for i in shard_idx]

    order = bucket_by_length(tokenizer, shard_texts, max_length=MAX_LENGTH)
    shard_texts = [shard_texts[i] for i in order]
    shard_rows  = [shard_rows[i]  for i in order]
    shard_rules = [shard_rules[i] for i in order]

    out_ids, out_probs, out_rules = [], [], []

    for i in range(0, len(shard_texts), BATCH_SIZE):
        batch_texts = shard_texts[i:i+BATCH_SIZE]
        enc = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
            pad_to_multiple_of=8,
            return_tensors="pt",
        )
        enc = {k: v.to(device, non_blocking=True) for k, v in enc.items()}

        with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.float16, enabled=(device.type == "cuda")):
            logits = model(**enc).logits  # (B,1)
            probs = torch.sigmoid(logits).squeeze(-1)  # (B,)

        out_probs.extend(probs.float().cpu().tolist())
        out_ids.extend(shard_rows[i:i+BATCH_SIZE])
        out_rules.extend(shard_rules[i:i+BATCH_SIZE])

    if world > 1:
        gi, gp, gr = [None]*world, [None]*world, [None]*world
        dist.all_gather_object(gi, out_ids)
        dist.all_gather_object(gp, out_probs)
        dist.all_gather_object(gr, out_rules)
        if rank == 0:
            all_ids   = [x for L in gi for x in L]
            all_p     = [x for L in gp for x in L]
            all_rules = [x for L in gr for x in L]

            out_df = pd.DataFrame({"row_id": all_ids, "rule": all_rules, "rule_violation": all_p})
            # Per-rule ranks scaled to [0,1]
            r = out_df.groupby("rule")["rule_violation"].rank(method="average", ascending=True)
            n = out_df.groupby("rule")["rule_violation"].transform("size")
            denom = (n - 1).where(n > 1, 1)
            out_df["rule_violation"] = (r - 1) / denom

            out_df[["row_id", "rule_violation"]].sort_values("row_id").to_csv("submission7.csv", index=False)
        dist.destroy_process_group()
    else:
        out_df = pd.DataFrame({"row_id": out_ids, "rule": out_rules, "rule_violation": out_probs})
        r = out_df.groupby("rule")["rule_violation"].rank(method="average", ascending=True)
        n = out_df.groupby("rule")["rule_violation"].transform("size")
        denom = (n - 1).where(n > 1, 1)
        out_df["rule_violation"] = (r - 1) / denom
        out_df[["row_id", "rule_violation"]].sort_values("row_id").to_csv("submission7.csv", index=False)

if __name__ == "__main__":
    main()