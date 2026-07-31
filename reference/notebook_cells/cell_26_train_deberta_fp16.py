import os, math, time
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ.setdefault("NCCL_IB_DISABLE", "1")
os.environ.setdefault("NCCL_ASYNC_ERROR_HANDLING", "1")

import torch
import torch.distributed as dist
from datetime import timedelta
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

from transformers import AutoTokenizer, AutoConfig, AutoModelForSequenceClassification, get_cosine_schedule_with_warmup

from utils import get_dataframe_to_train, build_classification_dataframe
from constants import DATA_PATH, LORA_PATH, BASE_MODEL_PATH

# Speed-friendly defaults
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.benchmark = True
if hasattr(torch, "set_float32_matmul_precision"):
    torch.set_float32_matmul_precision("high")

# -------- Config knobs (env) ----------
MAX_LENGTH = int(os.environ.get("MAX_LENGTH", "512"))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "8"))
EPOCHS     = int(os.environ.get("EPOCHS", "1"))
LR         = float(os.environ.get("LR", "2e-5"))
WD         = float(os.environ.get("WD", "0.01"))
WARMUP     = float(os.environ.get("WARMUP_RATIO", "0.00"))  # fraction of total steps
GRAD_ACCUM = int(os.environ.get("GRAD_ACCUM", "1"))
GRADIENT_CHECKPOINTING = os.environ.get("GRADIENT_CHECKPOINTING", "0") == "1"
# --------------------------------------

def setup_ddp():
    if "RANK" in os.environ:
        world = int(os.environ["WORLD_SIZE"])
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
        dist.init_process_group(
            backend="nccl" if torch.cuda.is_available() else "gloo",
            init_method="env://",
            timeout=timedelta(minutes=30)
        )
    else:
        world, rank, local_rank = 1, 0, 0
        if torch.cuda.is_available():
            torch.cuda.set_device(0)
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    return world, rank, local_rank, device

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

def main():
    world, rank, local_rank, device = setup_ddp()

    # Data
    raw_df = get_dataframe_to_train(DATA_PATH)

    # Tokenizer first, to know SEP token for concatenation
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH, use_fast=True, trust_remote_code=False)
    sep_token = tokenizer.sep_token if tokenizer.sep_token is not None else "</s>"

    df = build_classification_dataframe(raw_df, tok_sep_token=sep_token)

    train_ds = TxtClsDataset(df["text"], df["rule_violation"])
    if world > 1:
        sampler = DistributedSampler(train_ds, num_replicas=world, rank=rank, shuffle=True)
        shuffle = False
    else:
        sampler = None
        shuffle = True

    collate_fn = collate_fn_builder(tokenizer, MAX_LENGTH)
    loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        sampler=sampler,
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
    # Only full-params (no LoRA)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    total_steps = EPOCHS * len(loader) // max(1, GRAD_ACCUM)
    warmup_steps = int(WARMUP * total_steps)
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

    # Loss
    bce = torch.nn.BCEWithLogitsLoss()

    # AMP
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))
    model.train()

    if world > 1:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[local_rank] if device.type == "cuda" else None,
            output_device=local_rank if device.type == "cuda" else None,
            find_unused_parameters=False,
            broadcast_buffers=False
        )

    global_step = 0
    t0 = time.time()
    optimizer.zero_grad(set_to_none=True)

    for epoch in range(EPOCHS):
        if sampler is not None:
            sampler.set_epoch(epoch)
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

            if rank == 0 and global_step > 0 and global_step % 10 == 0:
                lr_now = scheduler.get_last_lr()[0]
                print(f"[epoch {epoch+1}] step {global_step}/{total_steps} | lr={lr_now:.6e} | loss={running/10.0:.4f}")
                running = 0.0

        if rank == 0:
            print(f"Epoch {epoch+1} done in {(time.time()-t0)/60:.1f} min")

    # Save (rank 0)
    if rank == 0:
        os.makedirs(LORA_PATH, exist_ok=True)  # reuse var name for output dir
        to_save = model.module if hasattr(model, "module") else model
        to_save.save_pretrained(LORA_PATH)
        tokenizer.save_pretrained(LORA_PATH)
        print(f"Saved model + tokenizer to {LORA_PATH}")

    if world > 1:
        dist.barrier()
        dist.destroy_process_group()

if __name__ == "__main__":
    main()