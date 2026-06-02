"""
삼성역을 중심으로 실제 거리뷰 기반 5km x 5km 맵 생성
- OpenStreetMap 데이터 활용
- 건물, 도로, 장애물 포함
- UAV 시뮬레이션용 인터랙티브 지도
"""

import folium
import geopandas as gpd
from shapely.geometry import box
import numpy as np
import json
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Rectangle
import requests
from pathlib import Path

# 삼성역 좌표 (서울, 강남구)
SAMSUNG_STATION_LAT = 37.0576
SAMSUNG_STATION_LON = 127.0605

# 5km x 5km 맵 범위 (중심에서 2.5km씩)
MAP_SIZE_KM = 5
EARTH_RADIUS_KM = 6371

def latlon_to_km_offset(lat, lon, center_lat, center_lon):
    """위도/경도를 중심점으로부터의 km 오프셋으로 변환"""
    lat_km = (lat - center_lat) * 111.0
    lon_km = (lon - center_lon) * 111.0 * np.cos(np.radians(center_lat))
    return lon_km, lat_km

def get_bounding_box(center_lat, center_lon, size_km=5):
    """중심점으로부터 주어진 크기의 경계 박스 반환"""
    offset_km = size_km / 2
    
    # km을 도 단위로 변환
    lat_offset = offset_km / 111.0
    lon_offset = offset_km / (111.0 * np.cos(np.radians(center_lat)))
    
    bbox = {
        'north': center_lat + lat_offset,
        'south': center_lat - lat_offset,
        'east': center_lon + lon_offset,
        'west': center_lon - lon_offset
    }
    return bbox

def get_osm_data(bbox, element_type='building'):
    """Overpass API를 사용하여 OSM 데이터 조회"""
    overpass_url = "https://overpass-api.de/api/interpreter"
    
    if element_type == 'building':
        query = f"""
        [out:json];
        (way["building"]({bbox['south']},{bbox['west']},{bbox['north']},{bbox['east']});
         relation["building"]({bbox['south']},{bbox['west']},{bbox['north']},{bbox['east']}););
        out geom;
        """
    elif element_type == 'highway':
        query = f"""
        [out:json];
        (way["highway"~"^(trunk|primary|secondary|tertiary|unclassified|residential)$"]
         ({bbox['south']},{bbox['west']},{bbox['north']},{bbox['east']}););
        out geom;
        """
    
    try:
        print(f"  OSM {element_type} 데이터 조회 중...")
        response = requests.get(overpass_url, params={'data': query}, timeout=30)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"  ⚠️  OSM 데이터 조회 실패: {e}")
        return None

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
            opacity=0.5
        ).add_to(m)
        
        # 세로선
        lon = bbox['west'] + i * lon_offset
        folium.PolyLine(
            [[bbox['south'], lon], [bbox['north'], lon]],
            color='gray',
            weight=1,
            opacity=0.5
        ).add_to(m)
    
    # OSM 데이터 조회 및 시각화
    print("\n[2/4] OSM 데이터 조회 중...")
    
    # 건물 데이터
    building_data = get_osm_data(bbox, 'building')
    if building_data and 'elements' in building_data:
        buildings = building_data['elements']
        for building in buildings:
            if 'geometry' in building:
                coords = [(node['lat'], node['lon']) for node in building['geometry']]
                if len(coords) >= 3:
                    folium.Polygon(
                        coords,
                        popup='건물',
                        color='orange',
                        fill=True,
                        fillColor='orange',
                        fillOpacity=0.5,
                        weight=1
                    ).add_to(m)
        print(f"  ✓ {len(buildings)}개 건물 추가됨")
    
    # 도로 데이터
    highway_data = get_osm_data(bbox, 'highway')
    if highway_data and 'elements' in highway_data:
        highways = highway_data['elements']
        for highway in highways:
            if 'geometry' in highway:
                coords = [(node['lat'], node['lon']) for node in highway['geometry']]
                if len(coords) >= 2:
                    folium.PolyLine(
                        coords,
                        popup='도로',
                        color='gray',
                        weight=3,
                        opacity=0.7
                    ).add_to(m)
        print(f"  ✓ {len(highways)}개 도로 추가됨")
    
    # 저장
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    m.save(output_path)
    print(f"  ✓ 인터랙티브 지도 저장: {output_path}")
    
    return m, bbox

def create_obstacle_grid(center_lat, center_lon, grid_size=50, output_path='models/samsung_obstacle_grid.npy'):
    """장애물 격자 생성 (0: 자유 공간, 1: 건물/도로)"""
    
    print("\n[3/4] 장애물 격자 생성 중...")
    
    bbox = get_bounding_box(center_lat, center_lon, MAP_SIZE_KM)
    
    # grid_size x grid_size 격자 생성
    obstacle_grid = np.zeros((grid_size, grid_size), dtype=np.uint8)
    
    # OSM 건물 데이터 조회
    building_data = get_osm_data(bbox, 'building')
    if building_data and 'elements' in building_data:
        buildings = building_data['elements']
        
        for building in buildings:
            if 'geometry' in building:
                coords = [(node['lat'], node['lon']) for node in building['geometry']]
                
                # 각 격자 셀에 대해 건물 내 포함 여부 확인
                for i in range(grid_size):
                    for j in range(grid_size):
                        # 격자 셀 중심 좌표
                        cell_lat = bbox['south'] + (i + 0.5) * (bbox['north'] - bbox['south']) / grid_size
                        cell_lon = bbox['west'] + (j + 0.5) * (bbox['east'] - bbox['west']) / grid_size
                        
                        # 간단한 포함 여부 확인 (격자 셀과 건물 경계의 겹침)
                        if is_point_in_polygon((cell_lat, cell_lon), coords):
                            obstacle_grid[i, j] = 1
    
    # OSM 도로 데이터는 별도로 처리 (도로는 통행 가능하지만 높이 제약)
    highway_data = get_osm_data(bbox, 'highway')
    if highway_data and 'elements' in highway_data:
        highways = highway_data['elements']
        for highway in highways:
            if 'geometry' in highway:
                coords = [(node['lat'], node['lon']) for node in highway['geometry']]
                # 도로 주변 2셀을 낮은 우선순위로 표시 (필요시)
    
    # 저장
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, obstacle_grid)
    print(f"  ✓ 장애물 격자 저장: {output_path} (크기: {grid_size}x{grid_size})")
    
    return obstacle_grid, bbox

def is_point_in_polygon(point, polygon):
    """Ray casting 알고리즘으로 점이 다각형 내에 있는지 확인"""
    lat, lon = point
    n = len(polygon)
    inside = False
    
    p1_lat, p1_lon = polygon[0]
    for i in range(1, n + 1):
        p2_lat, p2_lon = polygon[i % n]
        if lat > min(p1_lat, p2_lat):
            if lat <= max(p1_lat, p2_lat):
                if lon <= max(p1_lon, p2_lon):
                    if p1_lat != p2_lat:
                        xinters = (lat - p1_lat) * (p2_lon - p1_lon) / (p2_lat - p1_lat) + p1_lon
                    if p1_lon == p2_lon or lon <= xinters:
                        inside = not inside
        p1_lat, p1_lon = p2_lat, p2_lon
    
    return inside

def visualize_obstacle_grid(obstacle_grid, bbox, output_path='models/samsung_obstacle_grid.png'):
    """격자 시각화"""
    
    print("\n[4/4] 격자 시각화 중...")
    
    fig, ax = plt.subplots(figsize=(12, 12), dpi=100)
    
    # 격자 표시
    im = ax.imshow(obstacle_grid, cmap='RdYlGn_r', origin='lower', extent=[
        bbox['west'], bbox['east'], bbox['south'], bbox['north']
    ])
    
    # 삼성역 표시
    ax.plot(SAMSUNG_STATION_LON, SAMSUNG_STATION_LAT, 'r*', markersize=20, label='삼성역')
    
    # 5km x 5km 범위 표시
    rect = Rectangle(
        (bbox['west'], bbox['south']),
        bbox['east'] - bbox['west'],
        bbox['north'] - bbox['south'],
        linewidth=2,
        edgecolor='blue',
        facecolor='none',
        label='5km x 5km 범위'
    )
    ax.add_patch(rect)
    
    ax.set_xlabel('경도 (Longitude)')
    ax.set_ylabel('위도 (Latitude)')
    ax.set_title('삼성역 중심 5km x 5km 장애물 격자\n빨강: 건물/장애물, 초록: 자유 공간')
    ax.legend()
    
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('장애물 (0: 없음, 1: 있음)')
    
    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=100, bbox_inches='tight')
    print(f"  ✓ 격자 시각화 저장: {output_path}")
    plt.close()

def save_metadata(bbox, grid_size=50, output_path='models/samsung_map_metadata.json'):
    """메타데이터 저장"""
    metadata = {
        'center': {
            'latitude': SAMSUNG_STATION_LAT,
            'longitude': SAMSUNG_STATION_LON,
            'name': '삼성역'
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
            'resolution_m': int((MAP_SIZE_KM * 1000) / grid_size)
        },
        'data_source': 'OpenStreetMap (Overpass API)',
        'elements': ['buildings', 'highways'],
        'created_at': str(np.datetime64('now'))
    }
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    print(f"  ✓ 메타데이터 저장: {output_path}")

def main():
    print("=" * 60)
    print("삼성역 중심 5km x 5km 실제 거리뷰 기반 맵 생성")
    print("=" * 60)
    print(f"중심점: ({SAMSUNG_STATION_LAT}, {SAMSUNG_STATION_LON})")
    print(f"맵 크기: {MAP_SIZE_KM}km x {MAP_SIZE_KM}km")
    print(f"데이터 소스: OpenStreetMap (Overpass API)")
    
    # 1. 인터랙티브 지도 생성
    try:
        m, bbox = create_interactive_map(SAMSUNG_STATION_LAT, SAMSUNG_STATION_LON)
    except Exception as e:
        print(f"❌ 인터랙티브 지도 생성 실패: {e}")
        return
    
    # 2. 장애물 격자 생성
    try:
        obstacle_grid, bbox = create_obstacle_grid(SAMSUNG_STATION_LAT, SAMSUNG_STATION_LON, grid_size=50)
    except Exception as e:
        print(f"❌ 장애물 격자 생성 실패: {e}")
        return
    
    # 3. 격자 시각화
    try:
        visualize_obstacle_grid(obstacle_grid, bbox)
    except Exception as e:
        print(f"❌ 격자 시각화 실패: {e}")
        return
    
    # 4. 메타데이터 저장
    try:
        save_metadata(bbox)
    except Exception as e:
        print(f"❌ 메타데이터 저장 실패: {e}")
        return
    
    print("\n" + "=" * 60)
    print("✓ 맵 생성 완료!")
    print("생성된 파일:")
    print("  - models/samsung_station_map.html (인터랙티브 지도)")
    print("  - models/samsung_obstacle_grid.npy (장애물 격자)")
    print("  - models/samsung_obstacle_grid.png (격자 시각화)")
    print("  - models/samsung_map_metadata.json (메타데이터)")
    print("=" * 60)

if __name__ == "__main__":
    main()
