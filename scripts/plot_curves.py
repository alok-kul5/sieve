"""Plot train/val loss for every sweep run from results/curves/*.json -> docs/loss.png"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

COLORS = {"r8-lr2e-4": "#9aa0c7", "r16-lr1e-4": "#c7c2b4", "r16-lr2e-4": "#5967d6", "r32-lr2e-4": "#1626a8",
          "r32-lr2e-4-ep2": "#ff6a2b"}

fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), dpi=150)
for path in sorted(Path("results/curves").glob("*.json")):
    name, hist = path.stem, json.loads(path.read_text())
    c = COLORS.get(name, "#333")
    tr = [(h["epoch"], float(h["loss"])) for h in hist if "loss" in h]
    ev = [(h["epoch"], float(h["eval_loss"])) for h in hist if "eval_loss" in h]
    axes[0].plot(*zip(*tr), color=c, lw=2 if name.endswith("ep2") else 1.4, label=name)
    axes[1].plot(*zip(*ev), color=c, lw=2 if name.endswith("ep2") else 1.4, marker="o", ms=3, label=name)
for ax, t in zip(axes, ["Train loss (answer tokens only)", "Validation loss"]):
    ax.set_title(t, loc="left", fontsize=11, fontweight="bold")
    ax.set_xlabel("epoch")
    ax.set_yscale("log")
    ax.grid(alpha=.25)
    ax.spines[["top", "right"]].set_visible(False)
axes[1].legend(frameon=False, fontsize=8)
fig.tight_layout()
Path("docs").mkdir(exist_ok=True)
fig.savefig("docs/loss.png")
print("docs/loss.png")
