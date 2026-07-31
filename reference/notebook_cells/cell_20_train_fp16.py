import os, math, time
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ.setdefault("NCCL_IB_DISABLE", "1")
os.environ.setdefault("NCCL_ASYNC_ERROR_HANDLING", "1")

import torch
import torch.distributed as dist
from datetime import timedelta
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

from utils import get_dataframe_to_train, build_dataset
from constants import DATA_PATH, LORA_PATH, BASE_MODEL_PATH

# Speed-friendly defaults
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.benchmark = True
if hasattr(torch, "set_float32_matmul_precision"):
    torch.set_float32_matmul_precision("high")

# -------- Config knobs (env) ----------
# QUANT_MODE: "4bit" (default) or "8bit"
QUANT_MODE = os.environ.get("QUANT_MODE", "4bit").lower()
# For 4-bit only:
BNB_4BIT_TYPE = os.environ.get("BNB_4BIT_TYPE", "nf4")          # "nf4" or "fp4"
BNB_4BIT_DQ   = os.environ.get("BNB_4BIT_DQ", "1") == "1"       # double quant: 1/0
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

    if torch.cuda.is_available() and world > torch.cuda.device_count():
        raise RuntimeError(
            f"WORLD_SIZE={world} but only {torch.cuda.device_count()} visible CUDA device(s). "
            "Match --nproc_per_node to the number of GPUs or set CUDA_VISIBLE_DEVICES accordingly."
        )

    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    return world, rank, local_rank, device

class CompletionOnlyDataset(Dataset):
    def __init__(self, tokenizer, dataframe, max_length=256, add_eos_to_completion=True):
        self.tok = tokenizer
        self.max_length = max_length
        self.samples = []
        prompts = list(dataframe["prompt"])
        completions = list(dataframe["completion"])

        eos = self.tok.eos_token or ""
        for p, c in zip(prompts, completions):
            p_ids = self.tok.encode(p, add_special_tokens=False)
            c_text = c + (eos if add_eos_to_completion else "")
            c_ids = self.tok.encode(c_text, add_special_tokens=False)
            input_ids = p_ids + c_ids
            labels = [-100] * len(p_ids) + c_ids
            if len(input_ids) > self.max_length:
                excess = len(input_ids) - self.max_length
                input_ids = input_ids[excess:]
                labels = labels[excess:]
            self.samples.append({"input_ids": input_ids, "labels": labels})

    def __len__(self): return len(self.samples)
    def __getitem__(self, idx): return self.samples[idx]

def left_pad_collate(batch, pad_id, max_length=None):
    max_len = max(len(x["input_ids"]) for x in batch)
    if max_length is not None:
        max_len = min(max_len, max_length)
    input_ids, labels, attn = [], [], []
    for ex in batch:
        ids = ex["input_ids"][-max_len:]
        labs = ex["labels"][-max_len:]
        pad_len = max_len - len(ids)
        input_ids.append([pad_id]*pad_len + ids)
        labels.append([-100]*pad_len + labs)
        attn.append([0]*pad_len + [1]*len(ids))
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
        "attention_mask": torch.tensor(attn, dtype=torch.long),
    }

def count_trainable_params(model):
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total

def cosine_with_warmup_lambda(current_step, warmup_steps, total_steps):
    if current_step < warmup_steps:
        return float(current_step) / max(1, warmup_steps)
    progress = (current_step - warmup_steps) / max(1, total_steps - warmup_steps)
    return 0.5 * (1.0 + math.cos(math.pi * progress))

def main():
    world, rank, local_rank, device = setup_ddp()

    # Data
    df = get_dataframe_to_train(DATA_PATH)
    train_ds_hf = build_dataset(df)

    tok = AutoTokenizer.from_pretrained(BASE_MODEL_PATH, use_fast=True, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"

    # ---- Quantized base (bitsandbytes), fp16 compute only ----
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

    # Disable cache during training & prefer SDPA if available
    if hasattr(base, "config"):
        base.config.use_cache = False
        try:
            base.config._attn_implementation = "sdpa"
        except Exception:
            pass
    if getattr(base.config, "pad_token_id", None) is None:
        base.config.pad_token_id = tok.pad_token_id

    # Prepare for k-bit LoRA training (keeps fp16 compute)
    base = prepare_model_for_kbit_training(base, use_gradient_checkpointing=True)

    lora_cfg = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.1, bias="none",
        target_modules="all-linear",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(base, lora_cfg)

    if rank == 0:
        tr, tot = count_trainable_params(model)
        print(f"Trainable params: {tr:,} / {tot:,} ({100*tr/tot:.4f}%)")

    max_length = 256
    torch_ds = CompletionOnlyDataset(tok, train_ds_hf.to_pandas(), max_length=max_length)
    if world > 1:
        sampler = DistributedSampler(torch_ds, num_replicas=world, rank=rank, shuffle=True)
        shuffle = False
    else:
        sampler = None
        shuffle = True

    per_device_batch_size = 8
    loader = DataLoader(
        torch_ds,
        batch_size=per_device_batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=2,
        pin_memory=True,
        collate_fn=lambda b: left_pad_collate(b, pad_id=tok.pad_token_id, max_length=max_length),
        drop_last=False,
    )

    # Optimizer & scheduler (trainable params only)
    lr, wd = 1.5e-4, 0.01
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    try:
        optimizer = torch.optim.AdamW(
            trainable_params, lr=lr, weight_decay=wd,
            fused=(torch.cuda.is_available() and torch.__version__ >= "2.0")
        )
    except TypeError:
        optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=wd)

    num_epochs = 1
    steps_per_epoch = len(loader)
    total_steps = num_epochs * steps_per_epoch

    warmup_steps = 0
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda s: cosine_with_warmup_lambda(s, warmup_steps, total_steps)
    )

    # fp16 autocast (T4-safe; no bf16)
    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())
    model.train()

    if world > 1:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank] if torch.cuda.is_available() else None,
            output_device=local_rank if torch.cuda.is_available() else None,
            find_unused_parameters=False, broadcast_buffers=False
        )

    global_step = 0
    t0 = time.time()
    optimizer.zero_grad(set_to_none=True)

    for epoch in range(num_epochs):
        if sampler is not None:
            sampler.set_epoch(epoch)
        running = 0.0

        for step, batch in enumerate(loader):
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)

            with torch.cuda.amp.autocast(dtype=torch.float16, enabled=torch.cuda.is_available()):
                out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = out.loss

            scaler.scale(loss).backward()
            running += loss.item()

            scaler.unscale_(optimizer)
            clip_grad_norm_((p for p in model.parameters() if p.requires_grad), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()

            global_step += 1
            if rank == 0 and global_step % 10 == 0:
                avg = running / 10.0
                print(f"[epoch {epoch+1}] step {global_step}/{total_steps} | "
                      f"lr={scheduler.get_last_lr()[0]:.6e} | loss={avg:.4f}")
                running = 0.0

        if rank == 0:
            print(f"Epoch {epoch+1} done in {(time.time()-t0)/60:.1f} min")

    if rank == 0:
        os.makedirs(LORA_PATH, exist_ok=True)
        to_save = model.module if hasattr(model, "module") else model
        to_save.save_pretrained(LORA_PATH)
        tok.save_pretrained(LORA_PATH)
        print(f"Saved LoRA + tokenizer to {LORA_PATH}")

    if world > 1:
        dist.barrier()
        dist.destroy_process_group()

if __name__ == "__main__":
    main()