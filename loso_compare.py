#!/usr/bin/env python3
"""
loso_compare.py
───────────────
링크상태 예측: LSTM vs Transformer 의 일반화(Leave-One-Scenario-Out) 비교.

generalization_test.py 는 Transformer 만 측정했다. 여기서는 동일한 LOSO 절차로
LSTM 과 Transformer 를 둘 다 재학습·평가해 일반화 성능 차이를 본다.
시간 절약을 위해 대표 시나리오 5개에 대해 수행 (전부 비슷한 추세).

실행:
    python3 loso_compare.py
"""
from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import f1_score

import generalization_test as G   # load_csv, make_windows, DATA_DIR, LinkStateTransformer ...

# LSTM 정의 (pipeline.py와 동일)
class LinkStateLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(7, 64, 2, dropout=0.2, batch_first=True)
        self.dropout = nn.Dropout(0.2)
        self.head = nn.Linear(64, 3)
    def forward(self, x):
        o, _ = self.lstm(x)
        return self.head(self.dropout(o[:, -1, :]))


SUBSET = ["corridor_baseline", "cluster_spread", "relay_handover",
          "random_waypoint_1", "orbit"]


def train_loso(held_out, model_factory, device):
    all_rows = G.load_csv(G.DATA_DIR / "train.csv") + G.load_csv(G.DATA_DIR / "val.csv")
    train_rows = [r for r in all_rows if r["scenario_id"] != held_out]
    test_rows = (G.load_csv(G.DATA_DIR / "train.csv") + G.load_csv(G.DATA_DIR / "val.csv")
                 + G.load_csv(G.DATA_DIR / "test.csv"))
    test_rows = [r for r in test_rows if r["scenario_id"] == held_out]

    X_tr, y_tr = G.make_windows(train_rows)
    X_te, y_te = G.make_windows(test_rows)

    counts = np.bincount(y_tr, minlength=3)
    w = torch.tensor(len(y_tr) / (3.0 * counts), dtype=torch.float32).to(device)
    tr_loader = DataLoader(TensorDataset(torch.tensor(X_tr), torch.tensor(y_tr)),
                           batch_size=G.BATCH_SIZE, shuffle=True)
    model = model_factory().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=G.LR)
    loss_fn = nn.CrossEntropyLoss(weight=w)

    Xva = torch.tensor(X_tr[-len(X_tr)//5:]).to(device)
    yva = torch.tensor(y_tr[-len(y_tr)//5:]).to(device)
    best, patience, best_state = float("inf"), 0, None
    for ep in range(1, G.MAX_EPOCHS + 1):
        model.train()
        for xb, yb in tr_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad(); loss_fn(model(xb), yb).backward(); opt.step()
        model.eval()
        with torch.no_grad():
            vl = loss_fn(model(Xva), yva).item()
        if vl < best:
            best, patience = vl, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            patience += 1
            if patience >= G.PATIENCE:
                break
    model.load_state_dict(best_state); model.eval()
    with torch.no_grad():
        pred = model(torch.tensor(X_te).to(device)).argmax(1).cpu().numpy()
    return (pred == y_te).mean(), f1_score(y_te, pred, average="macro", zero_division=0)


def main():
    device = torch.device("mps" if torch.backends.mps.is_available()
                          else "cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\nLOSO 대상 {len(SUBSET)}개: {SUBSET}\n")
    factories = {"LSTM": LinkStateLSTM, "Transformer": G.LinkStateTransformer}
    res = {k: [] for k in factories}
    for sc in SUBSET:
        line = f"  held-out={sc:<20}"
        for name, fac in factories.items():
            acc, f1 = train_loso(sc, fac, device)
            res[name].append((acc, f1))
            line += f" | {name}: Acc={acc*100:5.1f}% F1={f1*100:5.1f}%"
        print(line)

    print(f"\n{'='*64}\n  LSTM vs Transformer — 링크상태 예측 일반화(LOSO)\n{'='*64}")
    print(f"  {'모델':<14}{'in-sample':>12}{'held-out(LOSO)':>16}{'일반화격차':>12}")
    print(f"  {'-'*52}")
    insample = {"LSTM": 93.83, "Transformer": 99.65}
    for name in factories:
        accs = np.array([a for a, _ in res[name]]) * 100
        ho = accs.mean()
        print(f"  {name:<14}{insample[name]:>11.2f}%{ho:>15.1f}%{ho-insample[name]:>+11.1f}%p")
    print(f"  {'-'*52}")


if __name__ == "__main__":
    main()
