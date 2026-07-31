import os
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
print("PID:", os.getpid(), "CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"))
import torch
assert torch.cuda.is_available()
print("Visible device count:", torch.cuda.device_count())
torch.cuda.set_device(0)            # 0 == the ONLY visible GPU in this process
print("Using:", torch.cuda.get_device_name(0))
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1") # no telemetry

import math
import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template, train_on_responses_only
from transformers import DataCollatorForSeq2Seq
from trl import SFTTrainer, SFTConfig
from typing import List, Tuple

from unsloth.chat_templates import CHAT_TEMPLATES

# =========================
# Merged CONSTANTS
# =========================
BASE_MODEL_PATH = os.environ.get("BASE_MODEL_PATH",
    "unsloth/Qwen3-14B-unsloth-bnb-4bit"
)
DATA_PATH = os.environ.get("DATA_PATH", "/kaggle/input/jigsaw-agile-community-rules/")

POSITIVE_ANSWER = "Yes"
NEGATIVE_ANSWER = "No"
BASE_PROMPT = f"Reddit moderation: Does the comment violate the rule? Answer '{POSITIVE_ANSWER}' or '{NEGATIVE_ANSWER}' only."

# Inference knobs
INFER_BATCH_SIZE = int(os.environ.get("INFER_BATCH_SIZE", 4))
WRITE_SUBMISSION = os.environ.get("WRITE_SUBMISSION", "1") == "1"  # write submission.csv

# =========================
# Merged UTILS
# =========================
def get_dataframe_to_train(data_path: str) -> pd.DataFrame:
    train_dataset = pd.read_csv(f"{data_path}/train.csv")
    test_dataset = pd.read_csv(f"{data_path}/test.csv")

    flatten = []

    # base train rows
    base = train_dataset[["body", "rule", "rule_violation"]].copy()
    base["source"] = "train"
    flatten.append(base)

    # Upsample target block (test examples) later by labeling them now
    for violation_type in ["positive", "negative"]:
        for i in range(1, 3):
            col = f"{violation_type}_example_{i}"
            sub_dataset = test_dataset[[col, "rule"]].copy()
            sub_dataset = sub_dataset.rename(columns={col: "body"})
            sub_dataset["rule_violation"] = 1 if violation_type == "positive" else 0
            sub_dataset["source"] = "test_examples"
            flatten.append(sub_dataset)

    # combine & dedupe first (so oversampling isn't undone)
    dataframe = pd.concat(flatten, axis=0, ignore_index=True)
    dataframe = dataframe.drop_duplicates(ignore_index=True)

    # upsample test_examples to appear 3x total (add two extra copies)
    test_rows = dataframe[dataframe["source"] == "test_examples"]
    if not test_rows.empty:
        dataframe = pd.concat([dataframe, test_rows], axis=0, ignore_index=True)

    # optional: shuffle for randomness
    dataframe = dataframe.sample(frac=1.0, random_state=1001).reset_index(drop=True)

    # drop helper column before returning
    dataframe = dataframe.drop(columns=["source"])
    return dataframe


# =========================
# Unsloth training helpers
# =========================
def load_dataframe() -> pd.DataFrame:
    df = get_dataframe_to_train(DATA_PATH)  # body, rule, rule_violation
    if "completion" not in df.columns:
        df = df.copy()
        df["completion"] = df["rule_violation"].map(
            {1: POSITIVE_ANSWER, 0: NEGATIVE_ANSWER}
        )
    return df[["body", "rule", "completion"]]

def make_conversations_dataset(df: pd.DataFrame) -> Dataset:
    # Each sample: system + user + assistant(Yes/No)
    convos = []
    for body, rule, comp in zip(df["body"], df["rule"], df["completion"]):
        convos.append([
            {"role": "system", "content": BASE_PROMPT},
            {"role": "user",   "content": f"Comment: {body}\n\nrule: {rule}"},
            {"role": "assistant", "content": str(comp)},
        ])
    return Dataset.from_dict({"conversations": convos})

def build_text_dataset(tokenizer, conv_dataset: Dataset):
    # Convert conversations -> single 'text' string via chat template.
    def formatting_prompts_func(examples):
        convos = examples["conversations"]
        texts = [
            tokenizer.apply_chat_template(
                conv, tokenize=False, add_generation_prompt=False, enable_thinking=False
            )[:-11]
            for conv in convos
        ]
        return {"text": texts}
    return conv_dataset.map(
        formatting_prompts_func,
        batched=True,
        remove_columns=conv_dataset.column_names,
    )

# =========================
# Inference helpers (Unsloth, in-memory)
# =========================
POSITIVE_VARIANTS = ["Yes", "YES", "Y", "yes", "True"]
NEGATIVE_VARIANTS = ["No",  "NO",  "N", "no",  "False"]
def _first_token_ids(tok, txt_or_texts) -> List[int]:
    """
    Return unique first-token IDs for one or many strings.
    Tries both the raw string and a space-prefixed variant.
    """
    texts = [txt_or_texts] if isinstance(txt_or_texts, str) else list(txt_or_texts)
    s = set()
    for t in texts:
        for t2 in (t, " " + t):
            ids = tok.encode(t2, add_special_tokens=False)
            if ids:
                s.add(ids[0])
    return sorted(s)

def _encode_batch(tok, bodies, rules, device, max_len=512):
    """
    Manual left truncation + left padding + explicit position_ids.
    Returns tensors on `device` with keys: input_ids, attention_mask, position_ids.
    """
    pad_id = tok.pad_token_id
    if pad_id is None:
        # Qwen templates usually set pad_token = eos_token already in your code,
        # but keep a safe fallback:
        pad_id = tok.eos_token_id if tok.eos_token_id is not None else 0

    # 1) Tokenize each sample via chat template
    seqs = []
    for body, rule in zip(bodies, rules):
        msgs = [
            {"role": "system", "content": BASE_PROMPT},
            {"role": "user",   "content": f"Comment: {body}\n\nrule: {rule}"},
        ]
        ids = tok.apply_chat_template(
            msgs,
            add_generation_prompt=True,
            tokenize=True,
            enable_thinking=False,
        )

        # Left truncation: keep the last `max_len` tokens
        if len(ids) > max_len:
            ids = ids[-max_len:]

        seqs.append(torch.tensor(ids, dtype=torch.long))

    # 2) Left padding to a uniform length (no longer than max_len)
    B = len(seqs)
    T = min(max_len, max(len(x) for x in seqs)) if seqs else 1

    input_ids      = torch.full((B, T), pad_id, dtype=torch.long)
    attention_mask = torch.zeros((B, T), dtype=torch.long)

    for i, ids in enumerate(seqs):
        L = min(T, len(ids))
        # Left pad: write actual ids into the *rightmost* part
        input_ids[i, T - L : T] = ids[-L:]
        attention_mask[i, T - L : T] = 1

    # 3) position_ids: first non-pad gets 0; pads use 0 (safe default)
    #    (Equivalent to: pos = cumsum(mask) - 1, then masked_fill(pad, 0))
    position_ids = attention_mask.cumsum(dim=1) - 1
    position_ids.masked_fill_(attention_mask.eq(0), 0)
    position_ids = position_ids.to(dtype=torch.long)

    # 4) Ship to device
    batch = {
        "input_ids":      input_ids.to(device, non_blocking=True),
        "attention_mask": attention_mask.to(device, non_blocking=True),
        "position_ids":   position_ids.to(device, non_blocking=True),
    }
    return batch


@torch.inference_mode()
def run_inference_unsloth_generate(
    model, tokenizer, data_path=DATA_PATH, batch_size=INFER_BATCH_SIZE,
    max_len=512, write_csv=WRITE_SUBMISSION, sort_by_length=True,
):
    tok = tokenizer

    yes_ids = _first_token_ids(tok, POSITIVE_VARIANTS)
    no_ids  = _first_token_ids(tok, NEGATIVE_VARIANTS)
    tgt_ids = sorted(set(yes_ids + no_ids))
    yes_idx = [tgt_ids.index(t) for t in yes_ids]
    no_idx  = [tgt_ids.index(t) for t in no_ids]

    # NOTE: this probably should be 'test.csv'; your current code reads 'train.csv'
    test_df = pd.read_csv(f"{data_path}/test.csv")
    bodies  = list(test_df["body"])
    rules   = list(test_df["rule"])
    rowids  = list(test_df["row_id"])

    N = len(bodies)
    if sort_by_length:
        approx_lens = [
            (len(tok.encode(b, add_special_tokens=False)) +
             len(tok.encode(r, add_special_tokens=False)))
            for b, r in zip(bodies, rules)
        ]
        sorted_idx = sorted(range(N), key=lambda i: min(approx_lens[i], max_len))
    else:
        sorted_idx = list(range(N))

    FastLanguageModel.for_inference(model)
    model.eval()
    device = next(model.parameters()).device

    probs_yes = [None] * N

    for i in range(0, N, batch_size):
        batch_indices = sorted_idx[i:i+batch_size]
        bb = [bodies[j] for j in batch_indices]
        rr = [rules[j]  for j in batch_indices]

        enc = _encode_batch(tok, bb, rr, device=device, max_len=max_len)

        out = model(**enc, use_cache=True)            # <— single forward pass
        step_scores = out.logits[:, -1, :]
        sel = step_scores[:, tgt_ids]
        logp = torch.log_softmax(sel.to(torch.float32), dim=-1)

        y_logp = (torch.logsumexp(logp[:, yes_idx], dim=-1)
                  if yes_idx else torch.full((sel.size(0),), -1e9, device=sel.device))
        n_logp = (torch.logsumexp(logp[:,  no_idx], dim=-1)
                  if no_idx  else torch.full((sel.size(0),), -1e9, device=sel.device))
        p_yes  = torch.softmax(torch.stack([y_logp, n_logp], dim=-1), dim=-1)[:, 0]

        for k, j in enumerate(batch_indices):
            probs_yes[j] = float(p_yes[k].item())

    # Build per-rule ranked scores in [0, 1]
    df_scores = pd.DataFrame({
        "row_id": rowids,
        "rule":   rules,
        "prob":   probs_yes,
    })

    grp   = df_scores.groupby("rule")
    rank  = grp["prob"].rank(method="average", ascending=True)   # low prob -> 1, high prob -> n
    n     = grp["prob"].transform("size")
    score = (rank - 1.0) / np.maximum(n - 1.0, 1.0)              # low -> 0.0, high -> 1.0
    df_scores["rule_violation"] = score

    out_df = df_scores[["row_id", "rule_violation"]].sort_values("row_id").reset_index(drop=True)
    if write_csv:
        out_df.to_csv("submission1.csv", index=False)
    return out_df


# =========================
# Main
# =========================
def main():
    # ---- Config knobs ----
    quant_mode = os.environ.get("QUANT_MODE", "4bit").lower()   # "4bit" or "8bit"
    load_in_4bit = (quant_mode == "4bit")
    dtype = None  # auto (fp16 on T4/V100, bf16 on Ampere+)

    # ---- Load base + LoRA via Unsloth ----
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name     = BASE_MODEL_PATH,
        max_seq_length = 512,
        dtype          = dtype,
        load_in_4bit   = load_in_4bit,
        load_in_8bit   = (quant_mode == "8bit"),
        local_files_only=True,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=["q_proj","k_proj","v_proj","o_proj",
                        "gate_proj","up_proj","down_proj"],
        lora_alpha=32,
        lora_dropout=0.0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=1001,
        use_rslora=False,
        loftq_config=None,
    )

    # ---- Tokenizer ----
    tokenizer = get_chat_template(tokenizer, chat_template="qwen-3")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.truncation_side = "left"

    # ---- Data ----
    df = load_dataframe()
    conv_dataset = make_conversations_dataset(df)
    train_dataset = build_text_dataset(tokenizer, conv_dataset)

    # ---- Trainer ----
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        dataset_text_field="text",
        max_seq_length=256,
        packing=False,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer),
        args=SFTConfig(
            per_device_train_batch_size=4,
            gradient_accumulation_steps=4,
            num_train_epochs=1,
            learning_rate=1.5e-4,
            weight_decay=0.01,
            lr_scheduler_type="linear",
            warmup_steps=0,
            logging_steps=10,
            optim="adamw_8bit",
            seed=1001,
            save_strategy="no",
            report_to="none",
            dataloader_num_workers=2,
        ),
    )

    # Only compute loss over assistant spans (our "Yes"/"No")
    trainer = train_on_responses_only(
        trainer,
        instruction_part = "<|im_start|>user",
        response_part = "<think>\n\n</think>\n\n",
    )

    trainer.train()

    # ---- Inference uses the SAME limit & left settings ----
    submission_df = run_inference_unsloth_generate(model, tokenizer, max_len=512)
    print(submission_df.head(10))
    print("Wrote submission.csv")

if __name__ == "__main__":
    main()