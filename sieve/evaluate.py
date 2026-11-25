"""Evaluation harness: run a model over the held-out test set and score it.

    # HF backends (GPU)
    python -m sieve.evaluate --backend hf --mode zero   --name base-zero
    python -m sieve.evaluate --backend hf --mode few    --name base-few
    python -m sieve.evaluate --backend hf --adapter runs/r16-lr2e-4/adapter --name finetuned
    # Ollama backend (GGUF)
    python -m sieve.evaluate --backend ollama --model sieve:q4_k_m --name gguf-q4
    # CI: no GPU, no model
    python -m sieve.evaluate --backend mock --test tests/fixtures/mini_test.jsonl --name mock
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import statistics
import time
from collections.abc import Callable
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

from .metrics import score
from .prompts import messages
from .schema import dumps

FEW_SHOT_IDS = 3


def load(path: str) -> list[dict]:
    return [json.loads(line) for line in Path(path).open(encoding="utf-8")]


def pick_shots(train: list[dict], k: int = FEW_SHOT_IDS, seed: int = 0) -> list[dict]:
    """Deterministic, varied shots: one INR, one without salary, one other currency."""
    rng = random.Random(seed)
    pools = [
        [r for r in train if r["label"]["salary"] and r["label"]["salary"]["currency"] == "INR"],
        [r for r in train if r["label"]["salary"] is None and r["label"]["company"] is None],
        [r for r in train if r["label"]["salary"] and r["label"]["salary"]["currency"] in ("EUR", "GBP")],
    ]
    shots = [rng.choice(p) for p in pools if p][:k]
    return shots or train[:k]


# ---------------------------------------------------------------- backends
def hf_backend(base: str, adapter: str | None, batch: int):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(base, padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(base, dtype=torch.bfloat16, device_map={"": 0},
                                                 low_cpu_mem_usage=True)
    if adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter).merge_and_unload()
    model.eval()
    torch.cuda.reset_peak_memory_stats()

    def run(prompts: list[list[dict]]) -> list[str]:
        outs = []
        for i in range(0, len(prompts), batch):
            chunk = [tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
                     for m in prompts[i:i + batch]]
            enc = tok(chunk, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
            with torch.no_grad():
                gen = model.generate(**enc, max_new_tokens=320, do_sample=False, temperature=None, top_p=None,
                                     top_k=None, pad_token_id=tok.pad_token_id)
            outs += tok.batch_decode(gen[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
        return outs

    def vram() -> float:
        return round(torch.cuda.max_memory_allocated() / 2**30, 2)

    return run, vram


def ollama_backend(model: str):
    import requests

    def run(prompts: list[list[dict]]) -> list[str]:
        outs = []
        for m in prompts:
            r = requests.post("http://localhost:11434/api/chat", timeout=600, json={
                "model": model, "messages": m, "stream": False, "keep_alive": "10m",
                "options": {"temperature": 0, "num_predict": 320, "num_ctx": 4096}})
            r.raise_for_status()
            outs.append(r.json()["message"]["content"])
        return outs

    def vram() -> float | None:
        try:
            ps = requests.get("http://localhost:11434/api/ps", timeout=10).json()["models"]
            hit = [x for x in ps if x["name"].startswith(model.split(":")[0])]
            return round(hit[0]["size_vram"] / 2**30, 2) if hit else None
        except Exception:
            return None

    return run, vram


def mock_backend(rows: list[dict]):
    """Deterministic fake model for CI: perfect on even rows, drops a skill on odd rows, garbage every 7th."""
    def run(prompts: list[list[dict]]) -> list[str]:
        outs = []
        for i, r in enumerate(rows[:len(prompts)]):
            lab = dict(r["label"])
            if i % 7 == 6:
                outs.append("Sure! Here is the JSON: {title: oops")
                continue
            if i % 2:
                lab["skills"] = lab["skills"][1:]
            outs.append(dumps(lab))
        return outs
    return run, lambda: 0.0


# ---------------------------------------------------------------- main
def evaluate(run: Callable, rows: list[dict], shots: list[dict], latency_n: int) -> tuple[list[str], dict]:
    prompts = [messages(r["text"], shots) for r in rows]
    lat = []
    if latency_n:
        run(prompts[:1])  # warm-up: load weights / build kernels outside the timed region
    for p in prompts[:latency_n]:  # single-request latency, as a user of the API would see it
        t = time.perf_counter()
        run([p])
        lat.append((time.perf_counter() - t) * 1000)
    t0 = time.perf_counter()
    outs = run(prompts)
    total = time.perf_counter() - t0
    timing = {"latency_ms_p50": round(statistics.median(lat), 1) if lat else None,
              "latency_ms_p95": round(sorted(lat)[max(0, math.ceil(0.95 * len(lat)) - 1)], 1) if lat else None,
              "throughput_per_s": round(len(rows) / total, 2) if total else None}
    return outs, timing


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["hf", "ollama", "mock"], required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--mode", choices=["zero", "few"], default="zero")
    ap.add_argument("--base", default="models/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--adapter")
    ap.add_argument("--model", help="ollama model tag")
    ap.add_argument("--test", default="data/test.jsonl")
    ap.add_argument("--train", default="data/train.jsonl")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--latency_n", type=int, default=20)
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    rows = load(args.test)[: args.limit or None]
    shots = pick_shots(load(args.train)) if args.mode == "few" else []
    if args.backend == "hf":
        run, vram = hf_backend(args.base, args.adapter, args.batch)
    elif args.backend == "ollama":
        run, vram = ollama_backend(args.model)
    else:
        run, vram = mock_backend(rows)
        args.latency_n = 0

    outs, timing = evaluate(run, rows, shots, args.latency_n)
    metrics = score(outs, [r["label"] for r in rows])
    result = {"name": args.name, "backend": args.backend, "mode": args.mode, "adapter": args.adapter,
              "model": args.model, **metrics, **timing, "peak_vram_gb": vram()}
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{args.name}.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    with (out / f"{args.name}.preds.jsonl").open("w", encoding="utf-8") as f:
        for r, o in zip(rows, outs):
            f.write(json.dumps({"id": r["id"], "output": o}, ensure_ascii=False) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "field_accuracy"}, indent=2))


if __name__ == "__main__":
    main()
