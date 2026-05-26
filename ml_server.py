"""
ml_server.py — NS-3 실시간 ML 연동 TCP 서버
NS-3가 1초마다 UAV 상태 JSON 전송 → 릴레이 노드 선택 + DQN 위치 보정 반환
"""

import json
import math
import os
import socket
import sys
import threading
import time
from collections import deque
from typing import Optional

# ── PyTorch / DQN + SB3 / PPO 로딩 ──────────────────────────────────────────
_TORCH_OK = False
_dqn_model = None
_ppo_model = None
try:
    import numpy as np
    import torch
    import torch.nn as nn
    _TORCH_OK = True
except ImportError:
    pass

try:
    from stable_baselines3 import PPO as SB3_PPO
    _SB3_OK = True
except ImportError:
    _SB3_OK = False

# 공통 물리 상수 (NS-3 와 동일)
COMM_RANGE   = 820.0    # m
RSSI_THRESH  = -85.0    # dBm
TX_POWER     = 20.0     # dBm
FREQ_MHZ     = 2400.0
NOISE_FLOOR  = -93.0    # dBm — 802.11g 20MHz, NF 8dB
# PPO 학습 환경과 동일한 상수
_PPO_RSSI_THRESH = -90.0   # rl_relay_agent.py 기준
_N_UAVS      = 5
_N_LINKS     = _N_UAVS * (_N_UAVS - 1) // 2   # 10

DIRECTIONS  = [i * (math.pi / 4) for i in range(8)]
# 5km 환경에 맞는 스텝 크기 (최대 400m)
STEP_SIZES  = [50.0, 100.0, 200.0, 400.0]
MAX_STEPS   = 20

MODEL_PATH     = os.path.join(os.path.dirname(__file__), "models", "rl_correction_dqn.pt")
PPO_MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "ppo_relay_agent.zip")


def _load_dqn():
    global _dqn_model
    if not _TORCH_OK:
        print("[DQN] PyTorch 없음 — 휴리스틱 폴백 사용")
        return
    if not os.path.exists(MODEL_PATH):
        print(f"[DQN] 모델 파일 없음: {MODEL_PATH}")
        return

    class _Net(nn.Module):
        def __init__(self, in_dim: int):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(in_dim, 128), nn.ReLU(),
                nn.Linear(128, 128), nn.ReLU(),
                nn.Linear(128, 32),
            )
        def forward(self, x):
            return self.net(x)

    try:
        sd = torch.load(MODEL_PATH, map_location="cpu")
        in_dim = sd["net.0.weight"].shape[1]   # 저장된 모델 입력 차원 자동 감지
        m = _Net(in_dim)
        m.load_state_dict(sd)
        m.eval()
        _dqn_model = m
        print(f"[DQN] 모델 로딩 완료: {MODEL_PATH} (입력={in_dim}D)")
    except Exception as e:
        print(f"[DQN] 로딩 실패: {e}")


def _load_ppo():
    global _ppo_model
    if not _SB3_OK:
        print("[PPO] stable-baselines3 없음 — RSSI 휴리스틱 폴백 사용")
        return
    if not os.path.exists(PPO_MODEL_PATH):
        print(f"[PPO] 모델 파일 없음: {PPO_MODEL_PATH} — RSSI 휴리스틱 폴백 사용")
        return
    try:
        _ppo_model = SB3_PPO.load(PPO_MODEL_PATH)
        print(f"[PPO] 모델 로딩 완료: {PPO_MODEL_PATH}")
    except Exception as e:
        print(f"[PPO] 로딩 실패: {e}")


# ── 설정 ──────────────────────────────────────────────────────────────────────
HOST = "127.0.0.1"
PORT = 9000

# 릴레이 선택 RSSI 임계 (NS-3 경로 손실 모델 기준)
NS3_RSSI_THRESH = -85.0   # dBm — 5km 시나리오 기준 (~820m 통신 범위)


# ── 전역 릴레이 상태 ─────────────────────────────────────────────────────────
_current_relay: int = -1   # _handle() 간 prev_relay 추적용

# ── 통계 ──────────────────────────────────────────────────────────────────────
_stats_lock = threading.Lock()
_stats = {
    "total": 0,
    "corrections": 0,
    "latencies_ms": deque(maxlen=200),
}


def log(msg: str):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ── RSSI / SNR / PLR 계산 유틸 ───────────────────────────────────────────────
def _rssi_fspl(ax: float, ay: float, bx: float, by: float) -> float:
    d = math.hypot(bx - ax, by - ay) or 0.01
    fspl = 20 * math.log10(d) + 20 * math.log10(FREQ_MHZ) - 27.55
    return TX_POWER - fspl


def _snr(rssi: float) -> float:
    return rssi - NOISE_FLOOR


def _plr(snr_db: float) -> float:
    import math as _m
    return 1.0 / (1.0 + _m.exp((snr_db - 10.0) * 0.5))


# ── 릴레이 선택: PPO 우선, 없으면 RSSI+SNR 휴리스틱 폴백 ────────────────────
def _build_ppo_state(uavs: list, links: list, prev_relay: int) -> "np.ndarray":
    """35D PPO 상태 벡터 구성 (RSSI×10 + SNR×10 + PLR×10 + relay_onehot×5)."""
    n = _N_UAVS
    link_map: dict = {}   # (src,dst) → {rssi, snr, plr}
    for lk in links:
        s, d = int(lk["src"]), int(lk["dst"])
        key = (min(s, d), max(s, d))
        link_map[key] = {
            "rssi": lk.get("rssi", -120.0),
            "snr":  lk.get("snr",  _snr(lk.get("rssi", -120.0))),
            "plr":  lk.get("plr",  _plr(_snr(lk.get("rssi", -120.0)))),
        }

    rssi_v, snr_v, plr_v = [], [], []
    for i in range(n):
        for j in range(i + 1, n):
            m = link_map.get((i, j), {"rssi": -120.0, "snr": -27.0, "plr": 1.0})
            rssi_v.append(float(np.clip((m["rssi"] - _PPO_RSSI_THRESH) / 50.0, -1.0, 1.0)))
            snr_v.append( float(np.clip((m["snr"]  - 10.0)             / 50.0, -1.0, 1.0)))
            plr_v.append( float(np.clip(m["plr"], 0.0, 1.0)))

    relay_oh = [0.0] * n
    if 0 <= prev_relay < n:
        relay_oh[prev_relay] = 1.0

    return np.array(rssi_v + snr_v + plr_v + relay_oh, dtype=np.float32)


def _select_relay(uavs: list, links: list, prev_relay: int = -1) -> int:
    # PPO 모델 있으면 추론
    if _ppo_model is not None and _TORCH_OK:
        try:
            state = _build_ppo_state(uavs, links, prev_relay)
            action, _ = _ppo_model.predict(state, deterministic=True)
            return int(action)
        except Exception as e:
            log(f"[PPO] 추론 실패, 휴리스틱 폴백: {e}")

    # 폴백: RSSI+SNR 가중 합산 휴리스틱
    score = {u["id"]: 0.0 for u in uavs}
    for lk in links:
        rssi = lk.get("rssi", -120.0)
        snr  = lk.get("snr", _snr(rssi))
        if rssi >= NS3_RSSI_THRESH:
            # SNR이 높을수록 가중치 증가 (단순 RSSI합 대비 개선)
            w = rssi + max(0.0, snr) * 0.5
            score[lk["src"]] += w
            score[lk["dst"]] += w
    n = len(uavs)
    return max(score, key=lambda k: (score[k], -(abs(k - n // 2))))


# ── DQN 상태 인코딩 (10D) ────────────────────────────────────────────────────
def _encode_state_with_main(iso_id: int, iso_x: float, iso_y: float,
                             main_pos: list,
                             link_data: Optional[dict] = None) -> "np.ndarray":
    """10D DQN 상태: 위치·거리·RSSI·SNR·PLR 포함."""
    if not main_pos:
        return np.zeros(10, dtype=np.float32)

    cx = sum(p[0] for p in main_pos) / len(main_pos)
    cy = sum(p[1] for p in main_pos) / len(main_pos)

    nearest = min(main_pos, key=lambda p: (p[0] - iso_x) ** 2 + (p[1] - iso_y) ** 2)
    nearest_d = math.hypot(nearest[0] - iso_x, nearest[1] - iso_y)

    nearest_rssi = _rssi_fspl(iso_x, iso_y, nearest[0], nearest[1])
    init_rssi    = max(_rssi_fspl(iso_x, iso_y, px, py) for px, py in main_pos)
    nearest_snr  = _snr(nearest_rssi)
    nearest_plr  = _plr(nearest_snr)

    return np.clip(np.array([
        (cx - iso_x) / COMM_RANGE,                   # 0: 중심 방향 x
        (cy - iso_y) / COMM_RANGE,                   # 1: 중심 방향 y
        math.hypot(cx - iso_x, cy - iso_y) / COMM_RANGE,  # 2: 중심 거리
        (nearest_rssi - RSSI_THRESH) / 30.0,         # 3: 최근접 RSSI
        nearest_d / COMM_RANGE,                      # 4: 최근접 거리
        0.0,                                         # 5: step_count
        0.0,                                         # 6: blocked
        (init_rssi - RSSI_THRESH) / 30.0,            # 7: 초기 RSSI
        (nearest_snr - 10.0) / 30.0,                 # 8: 최근접 SNR (NEW)
        nearest_plr,                                 # 9: 최근접 PLR (NEW)
    ], dtype=np.float32), -3.0, 3.0)


# ── DQN 위치 보정 ─────────────────────────────────────────────────────────────
def _compute_correction_dqn(uavs: list, links: list) -> Optional[dict]:
    """NS3 링크 RSSI 기반으로 고립 UAV를 찾아 DQN 보정 벡터 반환."""
    pos = {u["id"]: (u["x"], u["y"]) for u in uavs}

    # 각 UAV별 링크 수 / 최저 RSSI 집계
    link_rssi: dict[int, list] = {u["id"]: [] for u in uavs}
    for lk in links:
        link_rssi[lk["src"]].append(lk["rssi"])
        link_rssi[lk["dst"]].append(lk["rssi"])

    # 연결 그래프 구성: RSSI > 임계값인 링크만 연결됨
    n = len(uavs)
    adj: dict[int, set] = {u["id"]: set() for u in uavs}
    for lk in links:
        if lk["rssi"] >= NS3_RSSI_THRESH:
            adj[lk["src"]].add(lk["dst"])
            adj[lk["dst"]].add(lk["src"])

    # 연결 컴포넌트 탐색
    visited, comps = set(), []
    for uid in adj:
        if uid not in visited:
            comp, stack = set(), [uid]
            while stack:
                v = stack.pop()
                if v in visited:
                    continue
                visited.add(v); comp.add(v)
                stack.extend(adj[v] - visited)
            comps.append(comp)

    # 단일 컴포넌트 = 모두 연결됨 → 보정 불필요
    if len(comps) <= 1:
        return None

    # 가장 작은 컴포넌트의 UAV를 격리 UAV로 선정 (보통 1대)
    main_comp = max(comps, key=len)
    iso_comp  = min(comps, key=len)
    iso_id    = next(iter(iso_comp))

    ix, iy = pos[iso_id]
    # main_comp UAV 위치 리스트 (격리 UAV의 이동 목표 그룹)
    main_pos = [(pos[uid][0], pos[uid][1]) for uid in main_comp]

    if _dqn_model is not None:
        state = _encode_state_with_main(iso_id, ix, iy, main_pos)
        in_dim = _dqn_model.net[0].in_features
        with torch.no_grad():
            action = int(
                _dqn_model(torch.FloatTensor(state[:in_dim]).unsqueeze(0)).argmax().item()
            )
        direction = DIRECTIONS[action // len(STEP_SIZES)]
        dist_m    = STEP_SIZES[action % len(STEP_SIZES)]
        dx = dist_m * math.cos(direction)
        dy = dist_m * math.sin(direction)
    else:
        # 폴백: main_comp 무게중심 방향으로 10m 이동
        tx = sum(p[0] for p in main_pos) / len(main_pos)
        ty = sum(p[1] for p in main_pos) / len(main_pos)
        d  = math.hypot(tx - ix, ty - iy) or 1.0
        dx, dy = 10.0 * (tx - ix) / d, 10.0 * (ty - iy) / d

    return {"uav_id": iso_id, "dx": round(dx, 2), "dy": round(dy, 2)}


# ── 요청 처리 ──────────────────────────────────────────────────────────────────
def _handle(data: str) -> str:
    global _current_relay
    t0 = time.perf_counter()
    try:
        payload = json.loads(data)
    except json.JSONDecodeError as e:
        log(f"JSON 파싱 오류: {e}")
        return json.dumps({"relay": -1, "correction": None})

    uavs: list  = payload.get("uavs", [])
    links: list = payload.get("links", [])
    sim_t: float = payload.get("t", -1.0)

    relay_id   = _select_relay(uavs, links, prev_relay=_current_relay)
    _current_relay = relay_id
    correction = _compute_correction_dqn(uavs, links)

    elapsed_ms = (time.perf_counter() - t0) * 1000

    with _stats_lock:
        _stats["total"] += 1
        _stats["latencies_ms"].append(elapsed_ms)
        if correction:
            _stats["corrections"] += 1

    mode = "DQN" if (_dqn_model is not None and correction) else ("fallback" if correction else "ok")
    log(
        f"t={sim_t:.1f}s UAVs={len(uavs)} links={len(links)} "
        f"relay=UAV{relay_id} corr={correction} [{elapsed_ms:.2f}ms/{mode}]"
    )
    return json.dumps({"relay": relay_id, "correction": correction})


# ── 클라이언트 핸들러 ─────────────────────────────────────────────────────────
def _client_thread(conn: socket.socket, addr):
    log(f"NS-3 연결: {addr}")
    buf = b""
    try:
        while True:
            chunk = conn.recv(4096)
            if not chunk:
                break
            buf += chunk
            while b"\n" in buf:
                line, buf = buf.split(b"\n", 1)
                line = line.strip()
                if not line:
                    continue
                resp = _handle(line.decode("utf-8", errors="replace"))
                conn.sendall((resp + "\n").encode("utf-8"))
    except (ConnectionResetError, BrokenPipeError):
        pass
    except Exception as e:
        log(f"클라이언트 오류: {e}")
    finally:
        conn.close()
        log(f"연결 종료: {addr}")


# ── 통계 출력 스레드 ─────────────────────────────────────────────────────────
def _stats_printer():
    while True:
        time.sleep(10)
        with _stats_lock:
            total = _stats["total"]
            corr  = _stats["corrections"]
            lats  = list(_stats["latencies_ms"])
        avg = sum(lats) / len(lats) if lats else 0.0
        mx  = max(lats) if lats else 0.0
        log(f"[통계] 요청={total} 보정={corr} 평균={avg:.2f}ms 최대={mx:.2f}ms")


# ── 메인 ─────────────────────────────────────────────────────────────────────
def main():
    host, port = HOST, PORT
    if len(sys.argv) >= 3:
        host, port = sys.argv[1], int(sys.argv[2])

    _load_dqn()
    _load_ppo()

    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((host, port))
    srv.listen(5)
    relay_mode = "PPO" if _ppo_model else ("RSSI 휴리스틱" if not _ppo_model else "PPO")
    dqn_mode   = "DQN" if _dqn_model else "폴백"
    log(f"ML 서버 시작: {host}:{port} (릴레이={relay_mode} / 위치보정={dqn_mode})")

    threading.Thread(target=_stats_printer, daemon=True).start()

    try:
        while True:
            conn, addr = srv.accept()
            threading.Thread(target=_client_thread, args=(conn, addr), daemon=True).start()
    except KeyboardInterrupt:
        log("서버 종료")
    finally:
        srv.close()


if __name__ == "__main__":
    main()
