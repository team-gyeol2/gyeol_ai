"""
삼성역 5km x 5km 맵 사용 예제
"""

import numpy as np
import json
from pathlib import Path


def example1_load_map_data():
    """예제 1: 맵 데이터 로드"""
    print("=" * 60)
    print("예제 1: 맵 데이터 로드")
    print("=" * 60)
    
    # 1. 메타데이터 로드
    with open('models/samsung_map_metadata_v2.json', 'r') as f:
        metadata = json.load(f)
    
    print("\n[맵 정보]")
    print(f"중심점: {metadata['center']['name']}")
    print(f"위치: {metadata['center']['location']}")
    print(f"좌표: ({metadata['center']['latitude']}, {metadata['center']['longitude']})")
    print(f"맵 크기: {metadata['map_size_km']}km × {metadata['map_size_km']}km")
    print(f"격자 크기: {metadata['grid']['size']}×{metadata['grid']['size']}")
    print(f"셀 해상도: {metadata['grid']['resolution_m']}m/cell")
    
    # 2. 격자 데이터 로드
    grid = np.load('models/samsung_obstacle_grid_v2.npy')
    
    print(f"\n[격자 통계]")
    print(f"격자 크기: {grid.shape}")
    print(f"데이터 타입: {grid.dtype}")
    total_cells = grid.size
    obstacle_cells = np.sum(grid)
    free_cells = total_cells - obstacle_cells
    
    print(f"총 셀: {total_cells:,}개")
    print(f"장애물 셀: {obstacle_cells:,}개 ({obstacle_cells/total_cells*100:.1f}%)")
    print(f"자유 공간: {free_cells:,}개 ({free_cells/total_cells*100:.1f}%)")


def example2_coordinate_conversion():
    """예제 2: 좌표 변환"""
    print("\n\n" + "=" * 60)
    print("예제 2: 좌표 변환 (격자 <-> 위도/경도)")
    print("=" * 60)
    
    # 메타데이터 로드
    with open('models/samsung_map_metadata_v2.json', 'r') as f:
        metadata = json.load(f)
    
    bbox = metadata['bounding_box']
    grid_size = metadata['grid']['size']
    
    def grid_to_latlon(grid_i, grid_j):
        """격자 좌표를 위도/경도로 변환"""
        lat = bbox['south'] + (grid_i + 0.5) * (bbox['north'] - bbox['south']) / grid_size
        lon = bbox['west'] + (grid_j + 0.5) * (bbox['east'] - bbox['west']) / grid_size
        return lat, lon
    
    def latlon_to_grid(lat, lon):
        """위도/경도를 격자 좌표로 변환"""
        grid_i = int((lat - bbox['south']) / (bbox['north'] - bbox['south']) * grid_size)
        grid_j = int((lon - bbox['west']) / (bbox['east'] - bbox['west']) * grid_size)
        return max(0, min(grid_size-1, grid_i)), max(0, min(grid_size-1, grid_j))
    
    print(f"\n[격자 좌표 범위]")
    print(f"행(i): 0 ~ {grid_size-1}")
    print(f"열(j): 0 ~ {grid_size-1}")
    
    # 격자 좌표 예제
    test_positions = [
        (0, 0, "남서쪽 모서리"),
        (25, 25, "중앙 (삼성역 근처)"),
        (49, 49, "북동쪽 모서리"),
    ]
    
    print(f"\n[격자 → 위도/경도 변환]")
    for grid_i, grid_j, desc in test_positions:
        lat, lon = grid_to_latlon(grid_i, grid_j)
        print(f"  [{grid_i:2d}, {grid_j:2d}] ({desc:15s}) → {lat:.6f}°N, {lon:.6f}°E")
    
    # 위도/경도 예제
    print(f"\n[위도/경도 → 격자 변환]")
    test_coords = [
        (metadata['center']['latitude'], metadata['center']['longitude'], "삼성역"),
        (37.0576, 127.0605, "삼성역 (정확한 좌표)"),
        (37.0551, 127.0640, "강남역"),
    ]
    
    for lat, lon, desc in test_coords:
        grid_i, grid_j = latlon_to_grid(lat, lon)
        lat_check, lon_check = grid_to_latlon(grid_i, grid_j)
        print(f"  {lat:.6f}°N, {lon:.6f}°E ({desc:10s}) → [{grid_i:2d}, {grid_j:2d}]")
        print(f"    └─ 검증: {lat_check:.6f}°N, {lon_check:.6f}°E")


def example3_obstacle_detection():
    """예제 3: 장애물 감지"""
    print("\n\n" + "=" * 60)
    print("예제 3: 장애물 감지 및 안전 영역 찾기")
    print("=" * 60)
    
    # 격자 로드
    grid = np.load('models/samsung_obstacle_grid_v2.npy')
    
    def is_obstacle(grid_i, grid_j):
        """특정 위치가 장애물인지 확인"""
        if 0 <= grid_i < 50 and 0 <= grid_j < 50:
            return grid[grid_i, grid_j] == 1
        return True  # 범위 외는 장애물로 취급
    
    def is_safe_area(grid_i, grid_j, safety_radius=2):
        """안전 반경 내에 장애물이 없는지 확인"""
        for di in range(-safety_radius, safety_radius + 1):
            for dj in range(-safety_radius, safety_radius + 1):
                ni, nj = grid_i + di, grid_j + dj
                if is_obstacle(ni, nj):
                    return False
        return True
    
    print("\n[특정 위치의 장애물 확인]")
    test_positions = [
        (25, 25),
        (10, 10),
        (45, 45),
    ]
    
    for i, j in test_positions:
        obstacle = is_obstacle(i, j)
        status = "장애물 ⚠️" if obstacle else "안전 ✓"
        print(f"  [{i:2d}, {j:2d}]: {status}")
    
    print("\n[안전 영역 찾기 (반경 2셀)]")
    safe_count = 0
    for i in range(50):
        for j in range(50):
            if is_safe_area(i, j, safety_radius=2):
                safe_count += 1
    
    print(f"  안전 영역: {safe_count}/2500 ({safe_count/2500*100:.1f}%)")
    print(f"  추천 드론 이착륙 위치: {safe_count}개")
    
    # 연속된 안전 영역 찾기
    print("\n[연속된 안전 영역 분석]")
    visited = np.zeros_like(grid, dtype=bool)
    regions = []
    
    def bfs_safe_region(start_i, start_j):
        """안전한 격자로 이루어진 연결된 영역 찾기"""
        from collections import deque
        
        if visited[start_i, start_j] or is_obstacle(start_i, start_j):
            return 0
        
        queue = deque([(start_i, start_j)])
        visited[start_i, start_j] = True
        size = 0
        
        while queue:
            i, j = queue.popleft()
            size += 1
            
            # 4-근접성
            for di, dj in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                ni, nj = i + di, j + dj
                if 0 <= ni < 50 and 0 <= nj < 50 and not visited[ni, nj]:
                    if not is_obstacle(ni, nj):
                        visited[ni, nj] = True
                        queue.append((ni, nj))
        
        return size
    
    for i in range(50):
        for j in range(50):
            if not visited[i, j] and not is_obstacle(i, j):
                size = bfs_safe_region(i, j)
                if size > 0:
                    regions.append(size)
    
    regions.sort(reverse=True)
    print(f"  발견된 안전 영역: {len(regions)}개")
    if regions:
        print(f"  최대 영역 크기: {regions[0]} 셀 ({regions[0]*100*100:,}m²)")
        print(f"  평균 영역 크기: {np.mean(regions):.0f} 셀")


def example4_path_planning():
    """예제 4: 경로 계획 (간단한 버전)"""
    print("\n\n" + "=" * 60)
    print("예제 4: 경로 계획 (BFS 알고리즘)")
    print("=" * 60)
    
    grid = np.load('models/samsung_obstacle_grid_v2.npy')
    
    def find_path_bfs(grid, start, goal, max_steps=1000):
        """BFS 경로 탐색"""
        from collections import deque
        
        start_i, start_j = start
        goal_i, goal_j = goal
        
        # 그리드 크기 확인
        if grid[start_i, start_j] == 1 or grid[goal_i, goal_j] == 1:
            return None  # 시작이나 도착점이 장애물
        
        visited = np.zeros_like(grid, dtype=bool)
        parent = {}
        queue = deque([start])
        visited[start_i, start_j] = True
        parent[start] = None
        
        steps = 0
        while queue and steps < max_steps:
            steps += 1
            pos = queue.popleft()
            
            if pos == goal:
                # 경로 역추적
                path = []
                current = goal
                while current is not None:
                    path.append(current)
                    current = parent[current]
                return path[::-1]
            
            i, j = pos
            
            # 4-방향 탐색
            for di, dj in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                ni, nj = i + di, j + dj
                
                if 0 <= ni < 50 and 0 <= nj < 50 and not visited[ni, nj]:
                    if grid[ni, nj] == 0:  # 장애물 아님
                        visited[ni, nj] = True
                        parent[(ni, nj)] = pos
                        queue.append((ni, nj))
        
        return None  # 경로 없음
    
    # 경로 계획 예제
    start = (5, 5)
    goal = (45, 45)
    
    print(f"\n시작점: {start}")
    print(f"도착점: {goal}")
    
    path = find_path_bfs(grid, start, goal)
    
    if path:
        print(f"\n✓ 경로 찾음!")
        print(f"  경로 길이: {len(path)} 셀")
        
        # 실제 거리 계산 (100m/cell)
        actual_distance = len(path) * 100
        print(f"  실제 거리: {actual_distance}m ({actual_distance/1000:.1f}km)")
        
        # 직선 거리 (참고값)
        straight_line = np.sqrt((goal[0]-start[0])**2 + (goal[1]-start[1])**2) * 100
        print(f"  직선 거리: {straight_line:.0f}m")
        print(f"  경로 비율: {actual_distance/straight_line:.2f}x")
        
        # 비행 시간 계산 (속도 15m/s)
        flight_speed = 15  # m/s
        flight_time = actual_distance / flight_speed
        print(f"  비행 시간 (15m/s): {flight_time:.1f}초 ({flight_time/60:.1f}분)")
        
        # 경로 샘플 출력
        print(f"\n  경로 샘플 (처음 10개 포인트):")
        for idx, (i, j) in enumerate(path[:10]):
            print(f"    {idx:2d}. [{i:2d}, {j:2d}]")
        if len(path) > 10:
            print(f"    ...")
            print(f"    마지막 5개 포인트:")
            for idx, (i, j) in enumerate(path[-5:], len(path)-5):
                print(f"    {idx:2d}. [{i:2d}, {j:2d}]")
    else:
        print("\n✗ 경로를 찾을 수 없습니다!")


def example5_statistics():
    """예제 5: 격자 통계 분석"""
    print("\n\n" + "=" * 60)
    print("예제 5: 격자 통계 분석")
    print("=" * 60)
    
    grid = np.load('models/samsung_obstacle_grid_v2.npy')
    
    print("\n[통계 정보]")
    print(f"최소값: {np.min(grid)}")
    print(f"최대값: {np.max(grid)}")
    print(f"평균: {np.mean(grid):.4f}")
    print(f"표준편차: {np.std(grid):.4f}")
    
    # 행별 분석
    print("\n[행별 장애물 비율]")
    for i in range(50):
        obstacle_ratio = np.sum(grid[i, :]) / 50 * 100
        bar_len = int(obstacle_ratio / 5)
        bar = "█" * bar_len + "░" * (20 - bar_len)
        if i % 5 == 0:
            print(f"  행 {i:2d}: {bar} {obstacle_ratio:5.1f}%")
    
    # 열별 분석
    print("\n[열별 장애물 비율]")
    for j in range(0, 50, 5):
        obstacle_ratio = np.sum(grid[:, j]) / 50 * 100
        bar_len = int(obstacle_ratio / 5)
        bar = "█" * bar_len + "░" * (20 - bar_len)
        print(f"  열 {j:2d}: {bar} {obstacle_ratio:5.1f}%")


def main():
    """모든 예제 실행"""
    print("\n")
    print("╔" + "═" * 58 + "╗")
    print("║" + " " * 58 + "║")
    print("║" + "  삼성역 5km x 5km 맵 활용 예제".center(58) + "║")
    print("║" + " " * 58 + "║")
    print("╚" + "═" * 58 + "╝")
    
    try:
        example1_load_map_data()
        example2_coordinate_conversion()
        example3_obstacle_detection()
        example4_path_planning()
        example5_statistics()
        
        print("\n\n" + "=" * 60)
        print("✓ 모든 예제 완료!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ 에러 발생: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
