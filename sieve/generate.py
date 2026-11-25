"""Generate raw synthetic job posts with a local model.

    # in-process, 4-bit, batched (default)
    python -m sieve.generate --n 1800 --writer models/Phi-4-mini-instruct
    # or through a local Ollama server
    python -m sieve.generate --n 1800 --backend ollama --writer qwen2.5:7b
"""
from __future__ import annotations

import argparse
import json
import os
import random
import time
from dataclasses import asdict
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

from .synth import sample_spec, writer_prompt  # noqa: E402


def ollama_writer(model: str):
    import requests

    def write(prompts: list[str], seed: int) -> list[str]:
        outs = []
        for i, p in enumerate(prompts):
            for attempt in range(4):
                try:
                    r = requests.post("http://localhost:11434/api/generate", timeout=300, json={
                        "model": model, "prompt": p, "stream": False, "keep_alive": "30m",
                        "options": {"temperature": 0.9, "top_p": 0.95, "num_predict": 320, "seed": seed + i}})
                    r.raise_for_status()
                    outs.append(r.json()["response"].strip())
                    break
                except Exception:
                    time.sleep(5 * (attempt + 1))
            else:
                outs.append("")
        return outs
    return write


def hf_writer(path: str):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    tok = AutoTokenizer.from_pretrained(path, padding_side="left")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)
    model = AutoModelForCausalLM.from_pretrained(path, quantization_config=bnb, dtype=torch.bfloat16,
                                                 device_map={"": 0}, low_cpu_mem_usage=True).eval()

    def write(prompts: list[str], seed: int) -> list[str]:
        torch.manual_seed(seed)
        chats = [tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False,
                                         add_generation_prompt=True) for p in prompts]
        enc = tok(chats, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
        with torch.no_grad():
            gen = model.generate(**enc, max_new_tokens=300, do_sample=True, temperature=0.9, top_p=0.95,
                                 pad_token_id=tok.pad_token_id)
        return [t.strip() for t in tok.batch_decode(gen[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)]
    return write


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1800)
    ap.add_argument("--out", default="data/raw.jsonl")
    ap.add_argument("--backend", choices=["hf", "ollama"], default="hf")
    ap.add_argument("--writer", default="models/Phi-4-mini-instruct")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--batch", type=int, default=16)
    args = ap.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = sum(1 for _ in out.open(encoding="utf-8")) if out.exists() else 0
    rng = random.Random(args.seed)
    specs = [sample_spec(rng) for _ in range(args.n)]  # same seed -> same specs, so the run is resumable
    print(f"{done} already written, generating {args.n - done}", flush=True)
    write = hf_writer(args.writer) if args.backend == "hf" else ollama_writer(args.writer)
    name = Path(args.writer).name

    t0 = time.time()
    with out.open("a", encoding="utf-8") as f:
        for start in range(done, args.n, args.batch):
            chunk = specs[start:start + args.batch]
            texts = write([writer_prompt(s) for s in chunk], args.seed * 100_000 + start)
            for i, (spec, text) in enumerate(zip(chunk, texts)):
                rec = {"id": f"syn-{start + i:05d}", "writer": name, "style": spec.style, "text": text,
                       **{k: v for k, v in asdict(spec).items() if k != "style"}}
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            k = start + len(chunk) - done
            rate = k / (time.time() - t0)
            print(f"{start + len(chunk)}/{args.n}  {rate:.2f}/s  eta {(args.n - done - k) / rate / 60:.0f} min",
                  flush=True)


if __name__ == "__main__":
    main()
