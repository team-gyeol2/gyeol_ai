#!/usr/bin/env python3
"""학습 곡선 및 모델 비교 시각화"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = 'DejaVu Sans'

ROOT    = Path(__file__).resolve().parent
OUT_DIR = ROOT / "models"

# 히스토리 로드
with open(OUT_DIR / "lstm_history.json") as f:
    lstm_h = json.load(f)
with open(OUT_DIR / "transformer_history.json") as f:
    trans_h = json.load(f)

lstm_epochs  = [r["epoch"]      for r in lstm_h["history"]]
lstm_tloss   = [r["train_loss"] for r in lstm_h["history"]]
lstm_vloss   = [r["val_loss"]   for r in lstm_h["history"]]
lstm_vacc    = [r["val_acc"]    for r in lstm_h["history"]]

trans_epochs = [r["epoch"]      for r in trans_h["history"]]
trans_tloss  = [r["train_loss"] for r in trans_h["history"]]
trans_vloss  = [r["val_loss"]   for r in trans_h["history"]]
trans_vacc   = [r["val_acc"]    for r in trans_h["history"]]

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("LSTM vs Transformer — Training Curves", fontsize=14, fontweight="bold")

# ── 1. Train Loss ─────────────────────────────────────────────────────────────
ax = axes[0]
ax.plot(lstm_epochs,  lstm_tloss,  label="LSTM",        color="#2196F3", linewidth=2)
ax.plot(trans_epochs, trans_tloss, label="Transformer",  color="#FF5722", linewidth=2, linestyle="--")
ax.set_title("Train Loss")
ax.set_xlabel("Epoch")
ax.set_ylabel("Loss")
ax.legend()
ax.grid(True, alpha=0.3)

# ── 2. Val Loss ───────────────────────────────────────────────────────────────
ax = axes[1]
ax.plot(lstm_epochs,  lstm_vloss,  label="LSTM",        color="#2196F3", linewidth=2)
ax.plot(trans_epochs, trans_vloss, label="Transformer",  color="#FF5722", linewidth=2, linestyle="--")
ax.set_title("Validation Loss")
ax.set_xlabel("Epoch")
ax.set_ylabel("Loss")
ax.legend()
ax.grid(True, alpha=0.3)

# ── 3. 최종 성능 비교 Bar chart ───────────────────────────────────────────────
ax = axes[2]
models    = ["LSTM", "Transformer"]
test_accs = [lstm_h["test_acc"] * 100, trans_h["test_acc"] * 100]
colors    = ["#2196F3", "#FF5722"]
bars = ax.bar(models, test_accs, color=colors, width=0.4, alpha=0.85)
ax.set_title("Test Accuracy Comparison")
ax.set_ylabel("Accuracy (%)")
ax.set_ylim(95, 101)
ax.grid(True, alpha=0.3, axis="y")
for bar, acc in zip(bars, test_accs):
    ax.text(bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.05,
            f"{acc:.2f}%", ha="center", va="bottom", fontweight="bold", fontsize=12)

plt.tight_layout()
save_path = OUT_DIR / "training_curves.png"
plt.savefig(save_path, dpi=150, bbox_inches="tight")
print(f"저장 완료: {save_path}")
plt.show()
