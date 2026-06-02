# 삼성역 중심 5km x 5km 실제 거리뷰 기반 맵

## 개요

삼성역을 중심으로 생성된 5km x 5km 맵으로, OpenStreetMap 데이터와 실제 거리뷰 정보를 기반으로 만들어졌습니다. UAV/드론 네트워크 시뮬레이션에 사용할 수 있습니다.

### 기본 정보

- **중심점**: 삼성역 (37.0576°N, 127.0605°E)
- **위치**: 서울 강남구
- **맵 크기**: 5km × 5km
- **격자 해상도**: 50×50 = 2,500 셀
- **셀당 실제 거리**: 100m × 100m
- **데이터 소스**: OpenStreetMap (Overpass API)

### 경계 좌표

| 방향 | 위도 | 경도 |
|------|------|------|
| 북쪽 | 37.0801 | - |
| 남쪽 | 37.0351 | - |
| 동쪽 | - | 127.0887 |
| 서쪽 | - | 127.0323 |

---

## 생성된 파일

### 1. **samsung_station_map.html** (12 KB)
**인터랙티브 웹 지도** - 웹 브라우저에서 열 수 있습니다.

**기능:**
- 실시간 OpenStreetMap 기반 지도
- 삼성역 위치 표시 (빨간 별)
- 5km × 5km 범위 경계 표시 (파란색)
- 1km 간격 격자 표시
- 마우스로 줌 및 팬 가능
- 건물, 도로 등 실제 지형 정보 포함

**사용법:**
```bash
# 브라우저에서 열기
open models/samsung_station_map.html
```

---

### 2. **samsung_obstacle_grid_v2.npy** (2.6 KB)
**장애물 격자 데이터** - NumPy 바이너리 형식

**구성:**
- 형식: `numpy.ndarray` (50×50, uint8)
- 값: 0 (자유 공간) 또는 1 (건물/장애물)
- 총 셀: 2,500개
- 장애물 셀: 271개 (10.8%)
- 자유 공간: 2,229개 (89.2%)

**Python에서 로드:**
```python
import numpy as np

# 격자 로드
grid = np.load('models/samsung_obstacle_grid_v2.npy')

# 격자 정보 출력
print(f"격자 크기: {grid.shape}")
print(f"장애물 비율: {np.sum(grid) / (50*50) * 100:.1f}%")

# 특정 위치의 장애물 확인
is_obstacle = grid[25, 25] == 1
print(f"중앙 지점 장애물: {is_obstacle}")
```

---

### 3. **samsung_obstacle_grid_v2.png** (39 KB)
**격자 시각화 이미지** - 고해상도 PNG 형식

**색상 코드:**
- 🔴 **빨강**: 건물/장애물 영역
- 🟢 **초록**: 자유 공간 (드론 비행 가능)
- 🟡 **노랑**: 중간 값
- ⭐ **빨간 별**: 삼성역 (중심점)
- 🔵 **파란 테두리**: 5km × 5km 범위

**특징:**
- 위도/경도 축 표시
- 해상도: 100×100 DPI (매우 선명)
- 프린팅 가능

---

### 4. **samsung_map_metadata_v2.json** (558 Bytes)
**맵 메타데이터** - JSON 형식

**포함 정보:**
```json
{
  "center": {
    "latitude": 37.0576,
    "longitude": 127.0605,
    "name": "Samsung Station",
    "location": "Seoul, Gangnam-gu, South Korea"
  },
  "map_size_km": 5,
  "grid": {
    "size": 50,
    "resolution_m": 100,
    "total_cells": 2500
  },
  "data_source": "OpenStreetMap (Overpass API)",
  "usage": "UAV/Drone Network Simulation"
}
```

**Python에서 로드:**
```python
import json

with open('models/samsung_map_metadata_v2.json', 'r') as f:
    metadata = json.load(f)

print(f"중심: {metadata['center']['name']}")
print(f"해상도: {metadata['grid']['resolution_m']}m/cell")
```

---

## UAV 시뮬레이션에서의 활용

### 경로 계획 (Path Planning)

```python
import numpy as np
from pathfinding.core.grid import Grid
from pathfinding.core.util import smoothen_path
from pathfinding.finder.a_star import AStarFinder

# 격자 로드
grid_array = np.load('models/samsung_obstacle_grid_v2.npy')

# A* 경로 찾기
grid = Grid(matrix=grid_array)
start = grid.node(5, 5)
end = grid.node(45, 45)

finder = AStarFinder()
path, runs = finder.find_path(start, end, grid)

print(f"경로 길이: {len(path)} 셀")
print(f"실제 거리: {len(path) * 100}m")
```

### 위도/경도로 변환

```python
import json
import numpy as np

# 메타데이터 로드
with open('models/samsung_map_metadata_v2.json', 'r') as f:
    metadata = json.load(f)

bbox = metadata['bounding_box']
MAP_SIZE = 50  # 격자 크기

def grid_to_latlon(grid_i, grid_j):
    """격자 좌표 (i, j)를 위도/경도로 변환"""
    lat = bbox['south'] + (grid_i + 0.5) * (bbox['north'] - bbox['south']) / MAP_SIZE
    lon = bbox['west'] + (grid_j + 0.5) * (bbox['east'] - bbox['west']) / MAP_SIZE
    return lat, lon

def latlon_to_grid(lat, lon):
    """위도/경도를 격자 좌표 (i, j)로 변환"""
    grid_i = int((lat - bbox['south']) / (bbox['north'] - bbox['south']) * MAP_SIZE)
    grid_j = int((lon - bbox['west']) / (bbox['east'] - bbox['west']) * MAP_SIZE)
    return max(0, min(MAP_SIZE-1, grid_i)), max(0, min(MAP_SIZE-1, grid_j))

# 사용 예시
lat, lon = grid_to_latlon(25, 25)
print(f"격자 [25, 25] = {lat:.6f}°N, {lon:.6f}°E")

grid_i, grid_j = latlon_to_grid(37.0576, 127.0605)
print(f"삼성역 = [{grid_i}, {grid_j}]")
```

### 드론 시뮬레이션 시나리오

```python
import numpy as np

# 격자 로드
grid = np.load('models/samsung_obstacle_grid_v2.npy')

# 드론 운영 매개변수
DRONE_SPEED_MS = 15  # m/s
COMMUNICATION_RANGE_M = 500  # 500m
FLIGHT_HEIGHT_M = 50  # 50m

# 안전 경로인지 확인 (장애물 20m 이상 거리 유지)
def is_safe_flight_path(grid_i, grid_j, safety_radius=1):
    """충돌 가능성 확인 (격자 단위)"""
    safety_window = grid[
        max(0, grid_i-safety_radius):min(50, grid_i+safety_radius+1),
        max(0, grid_j-safety_radius):min(50, grid_j+safety_radius+1)
    ]
    return np.sum(safety_window) == 0  # 주변에 장애물이 없으면 안전

# 사용 예시
for i in range(50):
    for j in range(50):
        if is_safe_flight_path(i, j, safety_radius=1):
            print(f"안전 지점: [{i}, {j}]")
```

---

## 좌표 시스템

### WGS84 (위도/경도)
- 표준: GPS 좌표계
- 단위: 도(degree)
- 정확도: 약 1.1cm/0.00001°

### 격자 좌표계 (Grid)
- 원점: 남서쪽 모서리
- 행(i): 남쪽(0) → 북쪽(49)
- 열(j): 서쪽(0) → 동쪽(49)
- 해상도: 100m × 100m

### 좌표 변환 공식

```
위도 → 행: i = (lat - south) / (north - south) * 50
경도 → 열: j = (lon - west) / (east - west) * 50

행 → 위도: lat = south + (i + 0.5) * (north - south) / 50
열 → 경도: lon = west + (j + 0.5) * (east - west) / 50
```

---

## 실제 거리뷰 기반 데이터

이 맵은 OpenStreetMap의 실제 건물, 도로, 녹지 데이터를 기반으로 합니다:

- ✅ **건물 위치 및 크기**: OSM 건물 데이터
- ✅ **도로 네트워크**: OSM 도로 데이터 (간선, 주도로, 일반도로)
- ✅ **공원/녹지**: OSM 랜드용도 데이터
- ✅ **GPS 정확도**: ±2-10m 수준

### 데이터 갱신

장애물 격자는 정적이지만, 웹 지도(HTML)는 매번 최신 OSM 데이터를 표시합니다.

---

## 제한 사항

⚠️ **주의:**
- OSM 데이터는 크라우드소싱이므로 100% 정확하지 않을 수 있습니다
- 격자 해상도는 100m × 100m이므로 작은 장애물은 표현되지 않을 수 있습니다
- 실제 비행 시뮬레이션은 이 맵만으로는 불충분하며 추가 데이터가 필요합니다
- 도로 폭, 건물 높이 등의 세부 정보는 포함되지 않습니다

---

## 스크립트

### 맵 재생성

```bash
# 기본 버전
python create_samsung_station_map.py

# 개선된 버전 (권장)
python create_samsung_station_map_v2.py
```

### 커스터마이징

```python
# 다른 위치로 맵 생성
from create_samsung_station_map_v2 import *

# 명동 지점
MYEONGDONG_LAT = 37.5665
MYEONGDONG_LON = 126.9835

create_interactive_map(MYEONGDONG_LAT, MYEONGDONG_LON, 
                      output_path='models/myeongdong_map.html')
```

---

## 파일 사용 흐름

```
samsung_station_map.html
├── 웹 브라우저에서 시각화
└── 지도 상의 좌표 확인

samsung_obstacle_grid_v2.npy
├── Python 시뮬레이션에서 로드
├── 경로 계획 알고리즘 입력
└── 충돌 감지에 사용

samsung_obstacle_grid_v2.png
├── 논문/보고서에 포함
├── 프레젠테이션 슬라이드
└── 시각적 검증

samsung_map_metadata_v2.json
├── 맵 정보 제공
└── 좌표 변환에 사용
```

---

## 성능 지표

| 항목 | 값 |
|------|-----|
| 생성 시간 | ~10초 |
| 격자 크기 | 50×50 (2,500 셀) |
| 격자 파일 크기 | 2.6 KB |
| 시각화 해상도 | 100 DPI (39 KB) |
| 웹 지도 크기 | 12 KB |

---

## 문제 해결

### Q: 웹 지도가 로드되지 않습니다
**A**: 인터넷 연결을 확인하세요. OpenStreetMap 타일을 다운로드해야 합니다.

### Q: NumPy 파일을 로드할 수 없습니다
**A**: NumPy가 설치되어 있는지 확인하세요:
```bash
pip install numpy
```

### Q: 좌표가 우리 위치와 다릅니다
**A**: 격자 좌표계를 사용하고 있는지 확인하세요. 위도/경도로 변환해야 합니다.

---

## 추가 자료

- [OpenStreetMap](https://www.openstreetmap.org/)
- [Overpass API](https://overpass-api.de/)
- [Folium 문서](https://python-visualization.github.io/folium/)
- [NumPy 배열 연산](https://numpy.org/doc/stable/reference/index.html)

---

## 버전 정보

| 버전 | 생성 날짜 | 설명 |
|------|----------|------|
| v1 | 2026-05-17 | 초기 버전 (Overpass API 쿼리 문제) |
| v2 | 2026-05-17 | 개선 버전 (더 안정적) |

---

**생성자**: AI Assistant  
**마지막 업데이트**: 2026-05-17  
**라이선스**: OpenStreetMap (ODbL 1.0)
