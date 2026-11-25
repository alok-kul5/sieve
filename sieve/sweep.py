"""Small hyperparameter sweep over LoRA rank and learning rate.

    python -m sieve.sweep --epochs 1

Each config: train (sieve.train) -> greedy decode the validation split -> score.
The winner (best val exact match, then fact F1) is copied to runs/best/adapter.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

GRID = [(8, 2e-4), (16, 2e-4), (16, 1e-4)]


def sh(args: list[str]) -> None:
    print("+", " ".join(args), flush=True)
    subprocess.run([sys.executable, "-m", *args], check=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", default="2")
    ap.add_argument("--only", default="", help="comma list of run names to (re)run")
    args = ap.parse_args()

    rows = []
    for rank, lr in GRID:
        name = f"r{rank}-lr{lr:.0e}".replace("e-0", "e-")
        run = Path("runs") / name
        if not args.only or name in args.only.split(","):
            if not (run / "adapter").exists():
                sh(["sieve.train", "--name", name, "--rank", str(rank), "--lr", str(lr), "--epochs", args.epochs])
            if not (Path("results/sweep") / f"{name}.json").exists():
                sh(["sieve.evaluate", "--backend", "hf", "--adapter", str(run / "adapter"), "--name", name,
                    "--test", "data/val.jsonl", "--out", "results/sweep", "--latency_n", "0"])
        if not (run / "summary.json").exists():
            continue
        tr = json.loads((run / "summary.json").read_text())
        ev = json.loads((Path("results/sweep") / f"{name}.json").read_text())
        rows.append({"name": name, "rank": rank, "lr": lr, "epochs": tr["epochs"], "val_loss": round(tr["val_loss"], 4),
                     "val_exact_match": ev["exact_match"], "val_fact_f1": round(ev["fact_f1"], 4),
                     "val_schema_valid": ev["schema_valid"], "train_minutes": tr["train_minutes"],
                     "peak_vram_gb": tr["peak_vram_gb"], "trainable_params": tr["trainable_params"]})

    rows.sort(key=lambda r: (r["val_exact_match"], r["val_fact_f1"]), reverse=True)
    Path("results").mkdir(exist_ok=True)
    Path("results/sweep.json").write_text(json.dumps(rows, indent=2))
    for r in rows:
        print(r)
    if rows:
        best = Path("runs") / rows[0]["name"] / "adapter"
        shutil.rmtree("runs/best", ignore_errors=True)
        shutil.copytree(best, "runs/best/adapter")
        Path("runs/best/NAME").write_text(rows[0]["name"])
        print("best:", rows[0]["name"])


if __name__ == "__main__":
    main()
