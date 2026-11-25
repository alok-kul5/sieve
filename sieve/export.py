"""Merge the LoRA adapter, convert to GGUF, and register quantized builds with Ollama.

    python -m sieve.export --adapter runs/best/adapter

Steps:
  1. base (bf16) + adapter -> merged HF checkpoint        export/merged/
  2. llama.cpp convert_hf_to_gguf.py -> f16 GGUF          export/sieve-f16.gguf
  3. llama-quantize -> q8_0 / q4_K_M GGUF, then `ollama create` from a Modelfile for each
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

from .prompts import SYSTEM

LLAMA_CPP = Path("third_party/llama.cpp")

MODELFILE = '''FROM {gguf}

TEMPLATE """{{{{- if .System }}}}<|im_start|>system
{{{{ .System }}}}<|im_end|>
{{{{ end }}}}{{{{- range .Messages }}}}<|im_start|>{{{{ .Role }}}}
{{{{ .Content }}}}<|im_end|>
{{{{ end }}}}<|im_start|>assistant
"""

SYSTEM """{system}"""

PARAMETER temperature 0
PARAMETER num_ctx 4096
PARAMETER stop <|im_end|>
PARAMETER stop <|endoftext|>
'''


def merge(base: str, adapter: str, out: Path) -> None:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model = AutoModelForCausalLM.from_pretrained(base, dtype=torch.bfloat16, low_cpu_mem_usage=True)
    model = PeftModel.from_pretrained(model, adapter).merge_and_unload()
    model.save_pretrained(out, safe_serialization=True)
    AutoTokenizer.from_pretrained(base).save_pretrained(out)


def to_gguf(merged: Path, out: Path) -> None:
    if not LLAMA_CPP.exists():
        subprocess.run(["git", "clone", "--depth", "1", "https://github.com/ggml-org/llama.cpp", str(LLAMA_CPP)],
                       check=True)
    env = {**os.environ, "PYTHONPATH": str(LLAMA_CPP / "gguf-py")}
    subprocess.run([sys.executable, str(LLAMA_CPP / "convert_hf_to_gguf.py"), str(merged), "--outfile", str(out),
                    "--outtype", "f16"], check=True, env=env)


def quantize(f16: Path, tag: str) -> Path:
    """llama.cpp's llama-quantize (build: cmake --build third_party/llama.cpp/build --target llama-quantize)."""
    if tag == "f16":
        return f16
    exe = next((LLAMA_CPP / "build" / "bin").glob("llama-quantize*"))
    out = f16.with_name(f16.name.replace("-f16", f"-{tag.lower()}"))
    extra = os.environ.get("LLAMA_RUNTIME_PATH", r"C:\msys64\ucrt64\bin")  # gcc runtime DLLs on Windows
    env = {**os.environ, "PATH": extra + os.pathsep + os.environ.get("PATH", "")}
    subprocess.run([str(exe), str(f16), str(out), tag], check=True, env=env)
    return out


def ollama_create(f16: Path, tags: list[str], name: str) -> None:
    for tag in tags:
        gguf = quantize(f16, tag)
        mf = gguf.with_suffix(".Modelfile")
        mf.write_text(MODELFILE.format(gguf=gguf.resolve().as_posix(), system=SYSTEM), encoding="utf-8")
        cmd = ["ollama", "create", f"{name}:{tag.lower()}", "-f", str(mf)]
        print(" ".join(cmd), flush=True)
        subprocess.run(cmd, check=True)
        print(f"{gguf.name}: {gguf.stat().st_size / 2**30:.2f} GB", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="models/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--adapter", default="")
    ap.add_argument("--name", default="sieve", help="ollama model name")
    ap.add_argument("--out", default="export")
    ap.add_argument("--quants", default="f16,q8_0,q4_K_M")
    ap.add_argument("--skip", default="", help="comma list of steps to skip: merge,gguf,ollama")
    args = ap.parse_args()

    out = Path(args.out) / args.name
    merged, gguf = out / "merged", out / f"{args.name}-f16.gguf"
    skip = set(filter(None, args.skip.split(",")))
    out.mkdir(parents=True, exist_ok=True)
    if "merge" not in skip:
        if args.adapter:
            merge(args.base, args.adapter, merged)
        else:  # base model export, for the side-by-side demo
            merged = Path(args.base)
    if "gguf" not in skip:
        to_gguf(merged, gguf)
    if "ollama" not in skip:
        ollama_create(gguf, args.quants.split(","), args.name)


if __name__ == "__main__":
    main()
