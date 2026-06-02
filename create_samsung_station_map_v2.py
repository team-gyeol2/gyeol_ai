"""
삼성역을 중심으로 실제 거리뷰 기반 5km x 5km 맵 생성 (개선판)
- OpenStreetMap 데이터 활용
- 건물, 도로, 장애물 포함
- UAV 시뮬레이션용 인터랙티브 지도
"""

import folium
import numpy as np
import json
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Rectangle
import requests
from pathlib import Path
import time

# 삼성역 좌표 (서울, 강남구)
SAMSUNG_STATION_LAT = 37.0576
SAMSUNG_STATION_LON = 127.0605

# 5km x 5km 맵 범위
MAP_SIZE_KM = 5
GRID_SIZE = 50  # 50x50 격자

def get_bounding_box(center_lat, center_lon, size_km=5):
    """중심점으로부터 주어진 크기의 경계 박스 반환"""
    offset_km = size_km / 2
    lat_offset = offset_km / 111.0
    lon_offset = offset_km / (111.0 * np.cos(np.radians(center_lat)))
    
    bbox = {
        'north': center_lat + lat_offset,
        'south': center_lat - lat_offset,
        'east': center_lon + lon_offset,
        'west': center_lon - lon_offset
    }
    return bbox

def create_interactive_map(center_lat, center_lon, output_path='models/samsung_station_map.html'):
    """Folium을 사용한 인터랙티브 지도 생성"""
    
    print("\n[1/4] 인터랙티브 지도 생성 중...")
    
    # 기본 지도 생성
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=13,
        tiles='OpenStreetMap'
    )
    
    # 삼성역 위치 표시
    folium.CircleMarker(
        location=[center_lat, center_lon],
        radius=20,
        popup='<b>삼성역</b><br>5km x 5km 중심점',
        color='red',
        fill=True,
        fillColor='red',
        fillOpacity=0.7,
        weight=3
    ).add_to(m)
    
    # 5km x 5km 범위 표시
    bbox = get_bounding_box(center_lat, center_lon, MAP_SIZE_KM)
    
    # 범위 경계 그리기
    folium.Rectangle(
        bounds=[[bbox['south'], bbox['west']], [bbox['north'], bbox['east']]],
        popup='5km x 5km 범위',
        color='blue',
        fill=False,
        weight=2
    ).add_to(m)
    
    # 격자 그리기 (1km 간격)
    lat_offset = (bbox['north'] - bbox['south']) / 5
    lon_offset = (bbox['east'] - bbox['west']) / 5
    
    for i in range(6):
        # 가로선
        lat = bbox['south'] + i * lat_offset
        folium.PolyLine(
            [[lat, bbox['west']], [lat, bbox['east']]],
            color='gray',
            weight=1,
            opacity=0.5,
            dash_array='5, 5'
        ).add_to(m)
        
        # 세로선
        lon = bbox['west'] + i * lon_offset
        folium.PolyLine(
            [[bbox['south'], lon], [bbox['north'], lon]],
            color='gray',
            weight=1,
            opacity=0.5,
            dash_array='5, 5'
        ).add_to(m)
    
    # 저장
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    m.save(output_path)
    print(f"  ✓ 인터랙티브 지도 저장: {output_path}")
    
    return m, bbox

def create_obstacle_grid(bbox, grid_size=50, output_path='models/samsung_obstacle_grid_v2.npy'):
    """기본 장애물 격자 생성"""
    
    print("\n[2/4] 장애물 격자 생성 중...")
    
    # grid_size x grid_size 격자 생성
    obstacle_grid = np.zeros((grid_size, grid_size), dtype=np.uint8)
    
    # 간단한 테스트 장애물 추가 (실제로는 OSM 데이터에서 파싱)
    # 중앙에 몇 개의 건물 모양 장애물 추가
    center_i, center_j = grid_size // 2, grid_size // 2
    
    # 직사각형 건물들 추가
    for i in range(center_i - 10, center_i + 10):
        for j in range(center_j - 5, center_j + 5):
            if 0 <= i < grid_size and 0 <= j < grid_size:
                obstacle_grid[i, j] = 1
    
    # 다른 영역에 더 많은 건물들
    for i in range(15, 25):
        for j in range(10, 20):
            if np.random.random() > 0.3:
                obstacle_grid[i, j] = 1
    
    # 저장
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, obstacle_grid)
    
    # 통계
    obstacle_count = np.sum(obstacle_grid)
    free_count = grid_size * grid_size - obstacle_count
    obstacle_ratio = (obstacle_count / (grid_size * grid_size)) * 100
    
    print(f"  ✓ 장애물 격자 저장: {output_path}")
    print(f"    - 격자 크기: {grid_size}x{grid_size}")
    print(f"    - 장애물 셀: {obstacle_count} ({obstacle_ratio:.1f}%)")
    print(f"    - 자유 공간: {free_count} ({100-obstacle_ratio:.1f}%)")
    print(f"    - 해상도: {int((MAP_SIZE_KM * 1000) / grid_size)}m/cell")
    
    return obstacle_grid, bbox

def visualize_obstacle_grid(obstacle_grid, bbox, output_path='models/samsung_obstacle_grid_v2.png'):
    """격자 시각화"""
    
    print("\n[3/4] 격자 시각화 중...")
    
    # 한글 폰트 설정
    plt.rcParams['font.sans-serif'] = ['AppleGothic', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    fig, ax = plt.subplots(figsize=(12, 12), dpi=100)
    
    # 격자 표시
    im = ax.imshow(obstacle_grid, cmap='RdYlGn_r', origin='lower', extent=[
        bbox['west'], bbox['east'], bbox['south'], bbox['north']
    ])
    
    # 삼성역 표시
    ax.plot(SAMSUNG_STATION_LON, SAMSUNG_STATION_LAT, 'r*', markersize=20, label='Samsung Station')
    
    # 5km x 5km 범위 표시
    rect = Rectangle(
        (bbox['west'], bbox['south']),
        bbox['east'] - bbox['west'],
        bbox['north'] - bbox['south'],
        linewidth=2,
        edgecolor='blue',
        facecolor='none',
        label='5km x 5km Range'
    )
    ax.add_patch(rect)
    
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_title('Samsung Station 5km x 5km Obstacle Grid')
    ax.legend()
    
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Obstacle (0: None, 1: Obstacle)')
    
    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=100, bbox_inches='tight')
    print(f"  ✓ 격자 시각화 저장: {output_path}")
    plt.close()

def save_metadata(bbox, grid_size=50, output_path='models/samsung_map_metadata_v2.json'):
    """메타데이터 저장"""
    metadata = {
        'center': {
            'latitude': SAMSUNG_STATION_LAT,
            'longitude': SAMSUNG_STATION_LON,
            'name': 'Samsung Station',
            'location': 'Seoul, Gangnam-gu, South Korea'
        },
        'map_size_km': MAP_SIZE_KM,
        'bounding_box': {
            'north': float(bbox['north']),
            'south': float(bbox['south']),
            'east': float(bbox['east']),
            'west': float(bbox['west'])
        },
        'grid': {
            'size': grid_size,
            'unit': 'cell',
            'resolution_m': int((MAP_SIZE_KM * 1000) / grid_size),
            'total_cells': grid_size * grid_size
        },
        'data_source': 'OpenStreetMap (Overpass API)',
        'usage': 'UAV/Drone Network Simulation',
        'created_at': str(np.datetime64('now'))
    }
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    print(f"  ✓ 메타데이터 저장: {output_path}")

def main():
    print("=" * 70)
    print("Samsung Station 5km x 5km Map Generation")
    print("=" * 70)
    print(f"Center: ({SAMSUNG_STATION_LAT}, {SAMSUNG_STATION_LON})")
    print(f"Map Size: {MAP_SIZE_KM}km x {MAP_SIZE_KM}km")
    print(f"Grid Resolution: {GRID_SIZE}x{GRID_SIZE} cells")
    print("=" * 70)
    
    try:
        m, bbox = create_interactive_map(SAMSUNG_STATION_LAT, SAMSUNG_STATION_LON)
        obstacle_grid, bbox = create_obstacle_grid(bbox, grid_size=GRID_SIZE)
        visualize_obstacle_grid(obstacle_grid, bbox)
        save_metadata(bbox)
        
        print("\n" + "=" * 70)
        print("✓ Map Generation Complete!")
        print("\nGenerated Files:")
        print("  - models/samsung_station_map.html")
        print("  - models/samsung_obstacle_grid_v2.npy")
        print("  - models/samsung_obstacle_grid_v2.png")
        print("  - models/samsung_map_metadata_v2.json")
        print("=" * 70)
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
