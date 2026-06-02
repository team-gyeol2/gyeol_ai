#!/usr/bin/env python3
"""규칙기반 relay 선택 휴리스틱 수식을 발표용 PNG로 렌더링."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

# 한글 폰트 (macOS)
for cand in ["AppleSDGothicNeo", "AppleGothic", "NanumGothic", "Malgun Gothic"]:
    try:
        font_manager.findfont(cand, fallback_to_default=False)
        plt.rcParams["font.family"] = cand
        break
    except Exception:
        continue
plt.rcParams["axes.unicode_minus"] = False

NAVY = "#1f3a93"
fig = plt.figure(figsize=(11, 8.2))
fig.patch.set_facecolor("white")
ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off")

def T(x, y, s, size=15, color="#222", weight="normal", ha="left", math=False):
    ax.text(x, y, s, fontsize=size, color=color, weight=weight, ha=ha, va="center")

# 제목
ax.add_patch(plt.Rectangle((0.04, 0.90), 0.012, 0.055, color=NAVY))
T(0.07, 0.925, "규칙기반 Relay 선택 휴리스틱", size=22, color=NAVY, weight="bold")

# 메인 식
T(0.07, 0.83, r"$\mathrm{relay}^{*} \;=\; \mathrm{arg\,max}_{\,i\in\{0..4\}}\ \mathrm{Score}_{i}$",
  size=21, color="#111")

T(0.07, 0.745, r"$\mathrm{Score}_{i} \;=\; \sum_{f}\, w_{f}\,\cdot\,\tilde{x}_{i,f}$",
  size=20, color="#111")

# 단계 설명
T(0.07, 0.665, "① UAV $i$의 feature 평균  (i가 속한 링크 $L_i$ 평균):", size=14, color=NAVY, weight="bold")
T(0.10, 0.605, r"$x_{i,f} \;=\; \frac{1}{|L_i|}\sum_{(i,j)\in L_i}\mathrm{feature}_f(i,j)$", size=18)

T(0.07, 0.525, r"② min–max 정규화 (UAV들 간), INVERT feature는 $1-(\cdot)$:", size=14, color=NAVY, weight="bold")
T(0.10, 0.455, r"$\tilde{x}_{i,f} = \frac{x_{i,f}-\min_k x_{k,f}}{\max_k x_{k,f}-\min_k x_{k,f}}$"
               r"$\quad(\,f\notin\mathrm{INVERT}\,)$", size=18)
T(0.10, 0.395, r"$\tilde{x}_{i,f} = 1 - \frac{x_{i,f}-\min_k x_{k,f}}{\max_k x_{k,f}-\min_k x_{k,f}}$"
               r"$\quad(\,f\in\mathrm{INVERT}\,)$", size=18)

# 가중치 표
T(0.07, 0.305, "③ 가중치 $w_f$  (합 = 1.0)", size=14, color=NAVY, weight="bold")
rows = [
    ("RSSI", "0.20", "↑ 클수록 좋음"),
    ("SNR", "0.15", "↑ 클수록 좋음"),
    ("PLR", "0.20", "↓ INVERT"),
    ("Throughput", "0.15", "↑ 클수록 좋음"),
    ("Distance", "0.10", "↓ INVERT"),
    ("Hop count", "0.10", "↓ INVERT"),
    ("Blocked building", "0.10", "↓ INVERT"),
]
x0, x1, x2 = 0.10, 0.40, 0.55
ytop = 0.275
T(x0, ytop, "feature", size=13, weight="bold", color="#555")
T(x1, ytop, "$w_f$", size=13, weight="bold", color="#555")
T(x2, ytop, "방향", size=13, weight="bold", color="#555")
for k, (f, w, d) in enumerate(rows):
    y = ytop - 0.030 * (k + 1)
    inv = "INVERT" in d
    T(x0, y, f, size=13)
    T(x1, y, w, size=13, color=NAVY, weight="bold")
    T(x2, y, d, size=12, color="#c0392b" if inv else "#27ae60")

# 각주
T(0.07, 0.025, "INVERT = {PLR, Distance, Hop, Blocked} — 작을수록 점수↑   |   relay = argmax Score   (pipeline.py)",
  size=11, color="#777")

out = "models/heuristic_formula.png"
plt.savefig(out, dpi=160, bbox_inches="tight", facecolor="white")
print("저장:", out)
