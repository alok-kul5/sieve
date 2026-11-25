"""QLoRA fine-tune of a small instruct model on the extraction task.

    python -m sieve.train --name r16-lr2e-4 --rank 16 --lr 2e-4

Loss is computed on the assistant JSON only (prompt tokens are masked to -100).
Logs go to runs/<name>/tb (TensorBoard); the adapter is saved to runs/<name>/adapter.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, Trainer, TrainingArguments

from .prompts import messages
from .schema import dumps

BASE = "models/Qwen2.5-1.5B-Instruct"
TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def load_rows(path: str) -> list[dict]:
    return [json.loads(line) for line in Path(path).open(encoding="utf-8")]


def encode(tok, row: dict, max_len: int) -> dict:
    prompt = tok.apply_chat_template(messages(row["text"]), tokenize=False, add_generation_prompt=True)
    target = dumps(row["label"]) + tok.eos_token  # <|im_end|> for Qwen
    p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
    t_ids = tok(target, add_special_tokens=False)["input_ids"]
    ids = (p_ids + t_ids)[:max_len]
    labels = ([-100] * len(p_ids) + t_ids)[:max_len]
    return {"input_ids": ids, "labels": labels}


class Collator:
    def __init__(self, pad_id: int):
        self.pad_id = pad_id

    def __call__(self, feats: list[dict]) -> dict:
        n = max(len(f["input_ids"]) for f in feats)
        ids, labels, mask = [], [], []
        for f in feats:
            k = n - len(f["input_ids"])
            ids.append(f["input_ids"] + [self.pad_id] * k)
            labels.append(f["labels"] + [-100] * k)
            mask.append([1] * len(f["input_ids"]) + [0] * k)
        return {"input_ids": torch.tensor(ids), "labels": torch.tensor(labels),
                "attention_mask": torch.tensor(mask)}


def load_4bit(base: str):
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
                             bnb_4bit_compute_dtype=torch.bfloat16)
    return AutoModelForCausalLM.from_pretrained(base, quantization_config=bnb, dtype=torch.bfloat16,
                                                device_map={"": 0}, low_cpu_mem_usage=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--base", default=BASE)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--alpha", type=int, default=0, help="default 2*rank")
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--epochs", type=float, default=2)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--max_len", type=int, default=768)
    ap.add_argument("--train", default="data/train.jsonl")
    ap.add_argument("--val", default="data/val.jsonl")
    ap.add_argument("--limit", type=int, default=0, help="use only N train rows (smoke test)")
    args = ap.parse_args()

    out = Path("runs") / args.name
    out.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(args.base)
    train_rows, val_rows = load_rows(args.train), load_rows(args.val)
    if args.limit:
        train_rows, val_rows = train_rows[:args.limit], val_rows[:max(4, args.limit // 4)]
    train_ds = [encode(tok, r, args.max_len) for r in train_rows]
    val_ds = [encode(tok, r, args.max_len) for r in val_rows]

    torch.cuda.reset_peak_memory_stats()
    model = load_4bit(args.base)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    model.config.use_cache = False
    model = get_peft_model(model, LoraConfig(r=args.rank, lora_alpha=args.alpha or 2 * args.rank,
                                             lora_dropout=0.05, target_modules=TARGETS, task_type="CAUSAL_LM"))
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    os.environ["TENSORBOARD_LOGGING_DIR"] = str(out / "tb")  # transformers 5 reads the TB dir from env
    steps = math.ceil(len(train_ds) / (args.batch * args.accum)) * args.epochs
    warmup = max(1, int(0.05 * steps))
    targs = TrainingArguments(
        output_dir=str(out / "ckpt"), num_train_epochs=args.epochs, learning_rate=args.lr,
        per_device_train_batch_size=args.batch, per_device_eval_batch_size=args.batch,
        gradient_accumulation_steps=args.accum, lr_scheduler_type="cosine", warmup_steps=warmup,
        logging_steps=5, eval_strategy="steps", eval_steps=20, save_strategy="steps", save_steps=20, save_total_limit=1, bf16=True,
        optim="paged_adamw_8bit", report_to=["tensorboard"],
        dataloader_num_workers=0, seed=42, remove_unused_columns=False,
    )
    trainer = Trainer(model=model, args=targs, train_dataset=train_ds, eval_dataset=val_ds,
                      data_collator=Collator(tok.pad_token_id))
    t0 = time.time()
    ckpts = sorted((out / "ckpt").glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[1]))
    trainer.train(resume_from_checkpoint=str(ckpts[-1]) if ckpts else None)  # survive VRAM spikes on a shared GPU
    minutes = (time.time() - t0) / 60
    final = trainer.evaluate()
    model.save_pretrained(out / "adapter")
    tok.save_pretrained(out / "adapter")

    summary = {"name": args.name, "rank": args.rank, "alpha": args.alpha or 2 * args.rank, "lr": args.lr,
               "epochs": args.epochs, "train_rows": len(train_ds), "trainable_params": trainable,
               "val_loss": final["eval_loss"], "train_minutes": round(minutes, 1),
               "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2),
               "log_history": trainer.state.log_history}
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "log_history"}, indent=2))


if __name__ == "__main__":
    main()
