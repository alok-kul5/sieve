"""Side-by-side demo: base Qwen2.5-1.5B vs the Sieve fine-tune on the same post.

    SIEVE_BACKEND=ollama uvicorn app.server:app --port 8017   # GGUF builds in local Ollama
    SIEVE_BACKEND=hf     uvicorn app.server:app --port 8017   # one bf16 model, adapter toggled on/off
    SIEVE_BACKEND=mock   uvicorn app.server:app --port 8017   # no model, for tests and UI work
"""
from __future__ import annotations

import json
import os
import time
from functools import lru_cache
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from sieve.metrics import field_correct, parse
from sieve.prompts import messages
from sieve.schema import FIELDS, dumps

ROOT = Path(__file__).resolve().parents[1]
BACKEND = os.environ.get("SIEVE_BACKEND", "ollama")
BASE_TAG = os.environ.get("SIEVE_BASE_TAG", "sieve-base:q4_k_m")
FT_TAG = os.environ.get("SIEVE_FT_TAG", "sieve:q4_k_m")
ADAPTER = os.environ.get("SIEVE_ADAPTER", str(ROOT / "runs/best/adapter"))
BASE_DIR = os.environ.get("SIEVE_BASE", str(ROOT / "models/Qwen2.5-1.5B-Instruct"))

app = FastAPI(title="Sieve")
app.mount("/static", StaticFiles(directory=ROOT / "app/static"), name="static")


class ExtractIn(BaseModel):
    text: str = Field(min_length=10, max_length=6000)
    gold: dict | None = None  # when a test sample is used, score both sides against its label


# ---------------------------------------------------------------- backends
def _ollama(tag: str, text: str) -> str:
    import requests
    r = requests.post("http://localhost:11434/api/chat", timeout=180, json={
        "model": tag, "messages": messages(text), "stream": False, "keep_alive": "15m",
        "options": {"temperature": 0, "num_predict": 320}})
    r.raise_for_status()
    return r.json()["message"]["content"]


@lru_cache(maxsize=1)
def _hf():
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(BASE_DIR)
    model = AutoModelForCausalLM.from_pretrained(BASE_DIR, dtype=torch.bfloat16, device_map={"": 0})
    return tok, PeftModel.from_pretrained(model, ADAPTER).eval()


def _hf_run(text: str, finetuned: bool) -> str:
    import torch
    tok, model = _hf()
    prompt = tok.apply_chat_template(messages(text), tokenize=False, add_generation_prompt=True)
    enc = tok(prompt, return_tensors="pt", add_special_tokens=False).to(model.device)
    with torch.no_grad():
        if finetuned:
            gen = model.generate(**enc, max_new_tokens=320, do_sample=False)
        else:
            with model.disable_adapter():
                gen = model.generate(**enc, max_new_tokens=320, do_sample=False)
    return tok.decode(gen[0, enc["input_ids"].shape[1]:], skip_special_tokens=True)


def _mock(text: str, finetuned: bool) -> str:
    rec = {k: None for k in FIELDS}
    rec.update(title=text.split("\n")[0][:60], seniority="mid", skills=[])
    return dumps(rec) if finetuned else "Here is the extracted data:\n```json\n" + json.dumps(rec) + "\n```"


def run(text: str, finetuned: bool) -> str:
    if BACKEND == "mock":
        return _mock(text, finetuned)
    if BACKEND == "hf":
        return _hf_run(text, finetuned)
    return _ollama(FT_TAG if finetuned else BASE_TAG, text)


def side(text: str, finetuned: bool) -> dict:
    t = time.perf_counter()
    try:
        raw = run(text, finetuned)
    except Exception as e:  # surface model errors in the UI instead of a blank 500
        raise HTTPException(503, f"model backend unavailable: {e}") from e
    ms = (time.perf_counter() - t) * 1000
    obj, ok_json, ok_schema = parse(raw)
    strict = ok_json and raw.strip().startswith("{") and raw.strip().endswith("}")
    return {"raw": raw, "parsed": obj, "json_valid": ok_json, "schema_valid": ok_schema, "strict": strict,
            "latency_ms": round(ms)}


# ---------------------------------------------------------------- routes
@app.get("/")
def index():
    return FileResponse(ROOT / "app/static/index.html")


@app.get("/api/health")
def health():
    return {"ok": True, "backend": BACKEND, "base": BASE_TAG, "finetuned": FT_TAG}


@app.get("/api/samples")
def samples():
    path = ROOT / "data/test.jsonl"
    if not path.exists():
        path = ROOT / "tests/fixtures/mini_test.jsonl"
    rows = [json.loads(line) for line in path.open(encoding="utf-8")]
    picks = rows[:: max(1, len(rows) // 8)][:8]
    return [{"id": r["id"], "style": r["style"], "text": r["text"], "label": r["label"]} for r in picks]


@app.get("/api/results")
def results():
    path = ROOT / "results/summary.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else []


@app.post("/api/extract")
def extract(body: ExtractIn):
    base, ft = side(body.text, False), side(body.text, True)
    out = {"base": base, "finetuned": ft}
    if body.gold:
        for s in (base, ft):
            s["fields_correct"] = (field_correct(s["parsed"], body.gold) if s["parsed"]
                                   else {k: False for k in FIELDS})
    return out
