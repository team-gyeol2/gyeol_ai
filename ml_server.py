"""
ml_server.py — NS-3 실시간 ML 연동 TCP 서버
NS-3가 1초마다 UAV 상태 JSON을 전송 → 릴레이 노드 선택 + 위치 보정 반환
"""

import json
import math
import socket
import sys
import threading
import time
from collections import deque
from typing import Optional

# ── 설정 ──────────────────────────────────────────────────────────────────────
HOST = "127.0.0.1"
PORT = 9000

# 경로 손실 기반 RSSI 임계값 (dBm)
RSSI_THRESH = -85.0
# 릴레이 후보: RSSI 합이 최대인 노드를 릴레이로 선택
# 위치 보정: 통신 불가 링크가 생기면 해당 UAV를 이웃 방향으로 이동

# ── 통계 ──────────────────────────────────────────────────────────────────────
stats_lock = threading.Lock()
stats = {
    "total_requests": 0,
    "corrections_applied": 0,
    "latencies_ms": deque(maxlen=200),
}


def log(msg: str):
    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


# ── 릴레이 선택 로직 ──────────────────────────────────────────────────────────
def select_relay(uavs: list, links: list) -> int:
    """각 노드의 이웃 RSSI 합이 최대인 노드를 릴레이로 반환."""
    score = {u["id"]: 0.0 for u in uavs}
    for lk in links:
        rssi = lk["rssi"]
        if rssi >= RSSI_THRESH:
            score[lk["src"]] += rssi
            score[lk["dst"]] += rssi
    # 점수가 같으면 중간 인덱스 선호 (n//2)
    best = max(score, key=lambda k: (score[k], -(abs(k - len(uavs) // 2))))
    return best


# ── 위치 보정 로직 ──────────────────────────────────────────────────────────
def compute_correction(uavs: list, links: list) -> Optional[dict]:
    """
    RSSI 임계값 미달 링크 중 가장 나쁜 링크를 찾아
    해당 두 노드 중 더 고립된 쪽을 상대 방향으로 10m 이동.
    """
    n = len(uavs)
    pos = {u["id"]: (u["x"], u["y"]) for u in uavs}
    neighbor_count = {u["id"]: 0 for u in uavs}

    bad_links = []
    for lk in links:
        neighbor_count[lk["src"]] += 1
        neighbor_count[lk["dst"]] += 1
        if lk["rssi"] < RSSI_THRESH:
            bad_links.append(lk)

    if not bad_links:
        return None

    # RSSI가 가장 낮은 링크를 우선 처리
    worst = min(bad_links, key=lambda l: l["rssi"])
    src_id, dst_id = worst["src"], worst["dst"]

    # 이웃이 적은 쪽(더 고립된 쪽)을 이동
    mover = src_id if neighbor_count[src_id] <= neighbor_count[dst_id] else dst_id
    target = dst_id if mover == src_id else src_id

    mx, my = pos[mover]
    tx, ty = pos[target]
    dist = math.hypot(tx - mx, ty - my) or 1.0
    step = 10.0  # meters

    dx = step * (tx - mx) / dist
    dy = step * (ty - my) / dist

    return {"uav_id": mover, "dx": round(dx, 2), "dy": round(dy, 2)}


# ── 요청 처리 ──────────────────────────────────────────────────────────────────
def handle_request(data: str) -> str:
    t0 = time.perf_counter()
    try:
        payload = json.loads(data)
    except json.JSONDecodeError as e:
        log(f"JSON 파싱 오류: {e} | raw={data[:80]}")
        return json.dumps({"relay": -1, "correction": None})

    uavs: list = payload.get("uavs", [])
    links: list = payload.get("links", [])
    sim_t: float = payload.get("t", -1.0)

    relay_id = select_relay(uavs, links)
    correction = compute_correction(uavs, links)

    elapsed_ms = (time.perf_counter() - t0) * 1000

    with stats_lock:
        stats["total_requests"] += 1
        stats["latencies_ms"].append(elapsed_ms)
        if correction:
            stats["corrections_applied"] += 1

    resp = {"relay": relay_id, "correction": correction}
    log(
        f"t={sim_t:.1f}s | UAVs={len(uavs)} links={len(links)} "
        f"relay={relay_id} corr={correction} [{elapsed_ms:.2f}ms]"
    )
    return json.dumps(resp)


# ── 클라이언트 핸들러 (스레드) ──────────────────────────────────────────────
def client_thread(conn: socket.socket, addr):
    log(f"NS-3 연결됨: {addr}")
    buf = b""
    try:
        while True:
            chunk = conn.recv(4096)
            if not chunk:
                break
            buf += chunk
            # 개행 또는 완전한 JSON 오브젝트 단위로 처리
            while b"\n" in buf:
                line, buf = buf.split(b"\n", 1)
                line = line.strip()
                if not line:
                    continue
                response = handle_request(line.decode("utf-8", errors="replace"))
                conn.sendall((response + "\n").encode("utf-8"))
    except (ConnectionResetError, BrokenPipeError):
        pass
    except Exception as e:
        log(f"클라이언트 오류: {e}")
    finally:
        conn.close()
        log(f"연결 종료: {addr}")


# ── 통계 출력 스레드 ─────────────────────────────────────────────────────────
def stats_printer():
    while True:
        time.sleep(10)
        with stats_lock:
            total = stats["total_requests"]
            corr = stats["corrections_applied"]
            lats = list(stats["latencies_ms"])
        if lats:
            avg_lat = sum(lats) / len(lats)
            max_lat = max(lats)
        else:
            avg_lat = max_lat = 0.0
        log(
            f"[통계] 총요청={total} 보정={corr} "
            f"평균지연={avg_lat:.2f}ms 최대지연={max_lat:.2f}ms"
        )


# ── 메인 ─────────────────────────────────────────────────────────────────────
def main():
    host = HOST
    port = PORT
    if len(sys.argv) >= 3:
        host = sys.argv[1]
        port = int(sys.argv[2])

    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((host, port))
    srv.listen(5)
    log(f"ML 서버 시작: {host}:{port} — NS-3 연결 대기 중...")

    t = threading.Thread(target=stats_printer, daemon=True)
    t.start()

    try:
        while True:
            conn, addr = srv.accept()
            ct = threading.Thread(target=client_thread, args=(conn, addr), daemon=True)
            ct.start()
    except KeyboardInterrupt:
        log("서버 종료")
    finally:
        srv.close()


if __name__ == "__main__":
    main()
