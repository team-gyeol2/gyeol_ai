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

# ── PyTorch / DQN 로딩 ───────────────────────────────────────────────────────
_TORCH_OK = False
_dqn_model = None
try:
    import numpy as np
    import torch
    import torch.nn as nn
    _TORCH_OK = True
except ImportError:
    pass

# DQN 학습 때와 동일한 상수
COMM_RANGE  = 160.0
RSSI_THRESH = -90.0
TX_POWER    = -20.0
FREQ_MHZ    = 2400.0
DIRECTIONS  = [i * (math.pi / 4) for i in range(8)]
STEP_SIZES  = [10.0, 20.0, 30.0, 40.0]
MAX_STEPS   = 20

MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "rl_correction_dqn.pt")


def _load_dqn():
    global _dqn_model
    if not _TORCH_OK:
        print("[DQN] PyTorch 없음 — 휴리스틱 폴백 사용")
        return
    if not os.path.exists(MODEL_PATH):
        print(f"[DQN] 모델 파일 없음: {MODEL_PATH}")
        return

    class _Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(8, 128), nn.ReLU(),
                nn.Linear(128, 128), nn.ReLU(),
                nn.Linear(128, 32),
            )
        def forward(self, x):
            return self.net(x)

    try:
        m = _Net()
        m.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
        m.eval()
        _dqn_model = m
        print(f"[DQN] 모델 로딩 완료: {MODEL_PATH}")
    except Exception as e:
        print(f"[DQN] 로딩 실패: {e}")


# ── 설정 ──────────────────────────────────────────────────────────────────────
HOST = "127.0.0.1"
PORT = 9000

# 릴레이 선택 RSSI 임계 (NS-3 경로 손실 모델 기준)
NS3_RSSI_THRESH = -85.0


# ── 통계 ──────────────────────────────────────────────────────────────────────
_stats_lock = threading.Lock()
_stats = {
    "total": 0,
    "corrections": 0,
    "latencies_ms": deque(maxlen=200),
}


def log(msg: str):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ── RSSI 계산 (DQN 학습 환경 기준: FSPL 모델) ────────────────────────────────
def _rssi_fspl(ax: float, ay: float, bx: float, by: float) -> float:
    d = math.hypot(bx - ax, by - ay) or 0.01
    fspl = 20 * math.log10(d) + 20 * math.log10(FREQ_MHZ) - 27.55
    return TX_POWER - fspl


# ── 릴레이 선택 (이웃 RSSI 합 최대 노드) ────────────────────────────────────
def _select_relay(uavs: list, links: list) -> int:
    score = {u["id"]: 0.0 for u in uavs}
    for lk in links:
        if lk["rssi"] >= NS3_RSSI_THRESH:
            score[lk["src"]] += lk["rssi"]
            score[lk["dst"]] += lk["rssi"]
    n = len(uavs)
    best = max(score, key=lambda k: (score[k], -(abs(k - n // 2))))
    return best


# ── DQN 상태 인코딩 ──────────────────────────────────────────────────────────
def _encode_state_with_main(iso_id: int, iso_x: float, iso_y: float,
                             main_pos: list) -> "np.ndarray":
    """격리 UAV 위치 + main 그룹 위치 리스트로 8차원 DQN 상태 생성."""
    if not main_pos:
        return np.zeros(8, dtype=np.float32)

    cx = sum(p[0] for p in main_pos) / len(main_pos)
    cy = sum(p[1] for p in main_pos) / len(main_pos)

    nearest = min(main_pos, key=lambda p: (p[0] - iso_x) ** 2 + (p[1] - iso_y) ** 2)
    nearest_d = math.hypot(nearest[0] - iso_x, nearest[1] - iso_y)

    # FSPL 모델 기준 RSSI (DQN 학습 환경과 동일한 모델)
    nearest_rssi = _rssi_fspl(iso_x, iso_y, nearest[0], nearest[1])
    init_rssi    = max(_rssi_fspl(iso_x, iso_y, px, py) for px, py in main_pos)

    return np.clip(np.array([
        (cx - iso_x) / COMM_RANGE,
        (cy - iso_y) / COMM_RANGE,
        math.hypot(cx - iso_x, cy - iso_y) / COMM_RANGE,
        (nearest_rssi - RSSI_THRESH) / 30.0,
        nearest_d / COMM_RANGE,
        0.0,  # step_count=0 (보정 첫 스텝)
        0.0,  # blocked=0 (기본 시나리오 장애물 없음)
        (init_rssi - RSSI_THRESH) / 30.0,
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
        with torch.no_grad():
            action = int(
                _dqn_model(torch.FloatTensor(state).unsqueeze(0)).argmax().item()
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
    t0 = time.perf_counter()
    try:
        payload = json.loads(data)
    except json.JSONDecodeError as e:
        log(f"JSON 파싱 오류: {e}")
        return json.dumps({"relay": -1, "correction": None})

    uavs: list  = payload.get("uavs", [])
    links: list = payload.get("links", [])
    sim_t: float = payload.get("t", -1.0)

    relay_id   = _select_relay(uavs, links)
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

    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((host, port))
    srv.listen(5)
    log(f"ML 서버 시작: {host}:{port} ({'DQN' if _dqn_model else '휴리스틱'} 모드)")

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
