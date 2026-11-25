"""Collect results/*.json into results/summary.json and a markdown table for the README."""
import json
from pathlib import Path

ROWS = [  # key, label
    ("base-zero", "Qwen2.5-1.5B-Instruct, zero-shot"),
    ("base-few", "Qwen2.5-1.5B-Instruct, 3-shot"),
    ("qwen7b-zero", "Qwen2.5-7B-Instruct q4 (Ollama), zero-shot"),
    ("qwen7b-few", "Qwen2.5-7B-Instruct q4 (Ollama), 3-shot"),
    ("finetuned", "Sieve: 1.5B + LoRA (bf16)"),
    ("gguf-f16", "Sieve GGUF f16 (Ollama)"),
    ("gguf-q8_0", "Sieve GGUF q8_0 (Ollama)"),
    ("gguf-q4_k_m", "Sieve GGUF q4_K_M (Ollama)"),
]
HIGHLIGHT = "gguf-q4_k_m"


def main() -> None:
    out = []
    for key, label in ROWS:
        p = Path("results") / f"{key}.json"
        if not p.exists():
            continue
        r = json.loads(p.read_text())
        out.append({"key": key, "label": label, "highlight": key == HIGHLIGHT, **r})
    if out and not any(r["highlight"] for r in out):
        ft = [r for r in out if r["key"] == "finetuned"]
        (ft or out)[-1]["highlight"] = True
    Path("results/summary.json").write_text(json.dumps(out, indent=2))

    def pct(x):
        return "–" if x is None else f"{x * 100:.1f}%"

    lines = ["| Model | Exact match | Field F1 | Valid JSON (schema) | p50 latency | Throughput | Peak VRAM |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for r in out:
        name = f"**{r['label']}**" if r["key"].startswith(("finetuned", "gguf")) else r["label"]
        lat = "–" if r.get("latency_ms_p50") is None else f"{r['latency_ms_p50'] / 1000:.2f} s"
        thr = "–" if r.get("throughput_per_s") is None else f"{r['throughput_per_s']:.2f}/s"
        vram = "–" if r.get("peak_vram_gb") is None else f"{r['peak_vram_gb']:.1f} GB"
        lines.append(f"| {name} | {pct(r['exact_match'])} | {pct(r['fact_f1'])} | {pct(r['schema_valid'])} | {lat} "
                     f"| {thr} | {vram} |")
    fields = list(out[0]["field_accuracy"]) if out else []
    lines += ["", "| Field accuracy | " + " | ".join(r["key"] for r in out) + " |",
              "|---|" + "---:|" * len(out)]
    for f in fields:
        lines.append(f"| `{f}` | " + " | ".join(pct(r["field_accuracy"][f]) for r in out) + " |")
    Path("results/table.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
