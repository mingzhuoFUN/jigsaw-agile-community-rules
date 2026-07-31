import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch, torch.distributed as dist
import pandas as pd
from datetime import timedelta
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from utils import build_dataset
from constants import DATA_PATH, LORA_PATH, BASE_MODEL_PATH, POSITIVE_ANSWER, NEGATIVE_ANSWER

torch.backends.cuda.matmul.allow_tf32 = True
if hasattr(torch, "set_float32_matmul_precision"):
    torch.set_float32_matmul_precision("high")

# -------- Config knobs (env) ----------
QUANT_MODE = os.environ.get("QUANT_MODE", "4bit").lower()
BNB_4BIT_TYPE = os.environ.get("BNB_4BIT_TYPE", "nf4")
BNB_4BIT_DQ   = os.environ.get("BNB_4BIT_DQ", "1") == "1"
# --------------------------------------

# Variants to aggregate (same spirit as train.py)
POSITIVE_VARIANTS = ["Yes", "YES", "Y", "yes", "True"]
NEGATIVE_VARIANTS = ["No",  "NO",  "N", "no",  "False"]

def _first_token_ids(tok, txt_or_list):
    """
    Return unique first-token IDs for one or many strings.
    Tries both the raw string and a space-prefixed variant.
    """
    texts = [txt_or_list] if isinstance(txt_or_list, str) else list(txt_or_list)
    s = set()
    for t in texts:
        for t2 in (t, " " + t):
            ids = tok.encode(t2, add_special_tokens=False)
            if ids:
                s.add(ids[0])
    return sorted(s)

def bucket_by_length(tok, prompts, max_length=512):
    lens = tok(prompts, return_length=True, truncation=True, max_length=max_length)
    order = sorted(range(len(prompts)), key=lambda k: lens["length"][k], reverse=True)
    return order

def main():
    # DDP-style sharding
    if "RANK" in os.environ:
        world = int(os.environ["WORLD_SIZE"])
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl", timeout=timedelta(minutes=30))
    else:
        world, rank, local_rank = 1, 0, 0

    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    tok = AutoTokenizer.from_pretrained(BASE_MODEL_PATH, use_fast=True, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"

    # Quantized base (bnb), fp16 compute only
    if QUANT_MODE == "8bit":
        bnb_config = BitsAndBytesConfig(load_in_8bit=True)
        if rank == 0: print("Using bitsandbytes: 8-bit (LLM.int8)")
    else:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=BNB_4BIT_TYPE,
            bnb_4bit_use_double_quant=BNB_4BIT_DQ,
            bnb_4bit_compute_dtype=torch.float16,  # force fp16
        )
        if rank == 0: print(f"Using bitsandbytes: 4-bit ({BNB_4BIT_TYPE}, double_quant={BNB_4BIT_DQ})")

    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_PATH,
        trust_remote_code=False,
        low_cpu_mem_usage=True,
        quantization_config=bnb_config,
        device_map={"": local_rank} if torch.cuda.is_available() else None,
    )

    # Load LoRA *without* merging (keep quantized)
    model = PeftModel.from_pretrained(base, LORA_PATH)
    model.eval()

    # Prefer SDPA if available
    try:
        model.config._attn_implementation = "sdpa"
    except Exception:
        pass

    # --- Aggregate first-token IDs for Yes/No (matches train.py approach) ---
    yes_ids = _first_token_ids(tok, POSITIVE_VARIANTS)
    no_ids  = _first_token_ids(tok, NEGATIVE_VARIANTS)

    # Fallback to the constants if variants somehow mapped to nothing
    if not yes_ids:
        yes_ids = _first_token_ids(tok, POSITIVE_ANSWER)
    if not no_ids:
        no_ids = _first_token_ids(tok, NEGATIVE_ANSWER)

    tgt_ids = sorted(set(yes_ids + no_ids))
    yes_idx = [tgt_ids.index(t) for t in yes_ids]
    no_idx  = [tgt_ids.index(t) for t in no_ids]

    df = pd.read_csv(f"{DATA_PATH}/test.csv")
    ds = build_dataset(df)
    prompts = list(ds["prompt"])
    rows    = list(df["row_id"])
    rules   = list(df["rule"])

    idx = list(range(len(prompts)))
    shard_idx     = idx[rank::world]
    shard_prompts = [prompts[i] for i in shard_idx]
    shard_rows    = [rows[i]    for i in shard_idx]
    shard_rules   = [rules[i]   for i in shard_idx]

    order = bucket_by_length(tok, shard_prompts, max_length=512)
    shard_prompts = [shard_prompts[i] for i in order]
    shard_rows    = [shard_rows[i]    for i in order]
    shard_rules   = [shard_rules[i]   for i in order]

    out_ids, out_probs, out_rules = [], [], []
    bs = 16
    max_len = 512

    for i in range(0, len(shard_prompts), bs):
        batch_prompts = shard_prompts[i:i+bs]
        enc = tok(
            batch_prompts,
            padding=True,
            pad_to_multiple_of=8,
            truncation=True,
            max_length=max_len,
            return_tensors="pt",
        )
        enc = {k: v.to(device, non_blocking=True) for k, v in enc.items()}

        # fp16 autocast (T4-safe; no bf16)
        with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.float16, enabled=(device.type == "cuda")):
            logits = model(**enc).logits[:, -1, :]

            # Select only the Yes/No first-token columns and compute in float32 for stability
            sel = logits[:, tgt_ids].to(torch.float32)
            logp = torch.log_softmax(sel, dim=-1)

        # Aggregate log-probs across all "Yes" and all "No" variants
        if yes_idx:
            ylp = torch.logsumexp(logp[:, yes_idx], dim=-1)
        else:
            ylp = torch.full((logp.size(0),), -1e9, device=logp.device)

        if no_idx:
            nlp = torch.logsumexp(logp[:, no_idx], dim=-1)
        else:
            nlp = torch.full((logp.size(0),), -1e9, device=logp.device)

        prob_yes = torch.softmax(torch.stack([ylp, nlp], dim=-1), dim=-1)[:, 0].float().cpu().tolist()

        out_probs.extend(prob_yes)
        out_ids.extend(shard_rows[i:i+bs])
        out_rules.extend(shard_rules[i:i+bs])

    if world > 1:
        gi, gp, gr = [None]*world, [None]*world, [None]*world
        dist.all_gather_object(gi, out_ids)
        dist.all_gather_object(gp, out_probs)
        dist.all_gather_object(gr, out_rules)
        if rank == 0:
            all_ids   = [x for L in gi for x in L]
            all_p     = [x for L in gp for x in L]
            all_rules = [x for L in gr for x in L]

            # Per-rule ranks scaled to [0,1]
            out_df = pd.DataFrame({"row_id": all_ids, "rule": all_rules, "rule_violation": all_p})
            r = out_df.groupby("rule")["rule_violation"].rank(method="average", ascending=True)
            n = out_df.groupby("rule")["rule_violation"].transform("size")
            denom = (n - 1).where(n > 1, 1)  # avoid /0 for singletons
            out_df["rule_violation"] = (r - 1) / denom

            out_df[["row_id", "rule_violation"]].sort_values("row_id").to_csv("submission6.csv", index=False)
        dist.destroy_process_group()
    else:
        out_df = pd.DataFrame({"row_id": out_ids, "rule": out_rules, "rule_violation": out_probs})
        r = out_df.groupby("rule")["rule_violation"].rank(method="average", ascending=True)
        n = out_df.groupby("rule")["rule_violation"].transform("size")
        denom = (n - 1).where(n > 1, 1)
        out_df["rule_violation"] = (r - 1) / denom
        out_df[["row_id", "rule_violation"]].sort_values("row_id").to_csv("submission6.csv", index=False)

if __name__ == "__main__":
    main()