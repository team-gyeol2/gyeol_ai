"""
삼성역 맵 기반 UAV 네트워크 애니메이션 생성
"""

import numpy as np
import json
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Circle, Rectangle
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class UAVNetworkSimulation:
    def __init__(self, grid_size=50, grid_resolution_m=100):
        self.grid_size = grid_size
        self.resolution_m = grid_resolution_m
        
        with open('models/samsung_map_metadata_v2.json', 'r') as f:
            self.metadata = json.load(f)
        
        self.grid = np.load('models/samsung_obstacle_grid_v2.npy')
        
        self.num_uavs = 4
        self.communication_range = 150
        self.fps = 30
        self.total_duration = 30
        self.total_frames = self.fps * self.total_duration
        
        self.uav_initial_pos = [(10, 10), (15, 35), (25, 25), (40, 40)]
        self.uav_target_pos = [(45, 45), (10, 45), (25, 25), (45, 10)]
    
    def grid_to_meters(self, grid_i, grid_j):
        x_m = grid_j * self.resolution_m
        y_m = grid_i * self.resolution_m
        return x_m, y_m
    
    def get_uav_position(self, uav_id, frame):
        start_pos = np.array(self.uav_initial_pos[uav_id], dtype=float)
        target_pos = np.array(self.uav_target_pos[uav_id], dtype=float)
        progress = min(frame / self.total_frames, 1.0)
        
        if uav_id == 2:
            progress = min(frame / (self.total_frames * 0.3), 1.0)
        
        current_pos = start_pos + (target_pos - start_pos) * progress
        return current_pos
    
    def get_uav_status(self, uav_id, frame):
        failure_start_frame = int(self.fps * 20)
        
        if uav_id == 2:
            if frame < failure_start_frame:
                return 'healthy'
            elif frame < failure_start_frame + int(self.fps * 3):
                return 'degraded'
            else:
                return 'disconnected'
        
        return 'healthy'
    
    def calculate_distance(self, pos1, pos2):
        x1, y1 = self.grid_to_meters(pos1[0], pos1[1])
        x2, y2 = self.grid_to_meters(pos2[0], pos2[1])
        return np.sqrt((x2-x1)**2 + (y2-y1)**2)
    
    def get_link_status(self, uav1_id, uav2_id, frame):
        pos1 = self.get_uav_position(uav1_id, frame)
        pos2 = self.get_uav_position(uav2_id, frame)
        distance = self.calculate_distance(pos1, pos2)
        
        status1 = self.get_uav_status(uav1_id, frame)
        status2 = self.get_uav_status(uav2_id, frame)
        
        if distance > self.communication_range:
            return 'disconnected'
        
        if status1 == 'disconnected' or status2 == 'disconnected':
            return 'disconnected'
        
        if status1 == 'degraded' or status2 == 'degraded':
            return 'degraded'
        
        return 'healthy'
    
    def generate_frame_data(self, frame):
        data = {'frame': frame, 'time': frame / self.fps, 'uavs': [], 'links': []}
        
        for uav_id in range(self.num_uavs):
            pos = self.get_uav_position(uav_id, frame)
            status = self.get_uav_status(uav_id, frame)
            x_m, y_m = self.grid_to_meters(pos[0], pos[1])
            data['uavs'].append({
                'id': uav_id + 1, 'x': x_m, 'y': y_m,
                'status': status, 'grid_pos': pos.tolist(),
            })
        
        for i in range(self.num_uavs):
            for j in range(i+1, self.num_uavs):
                link_status = self.get_link_status(i, j, frame)
                data['links'].append({'from': i + 1, 'to': j + 1, 'status': link_status})
        
        return data

def create_animation_frames(simulation):
    print("\n[1/3] 애니메이션 프레임 생성 중...")
    frames_data = []
    for frame in range(simulation.total_frames):
        data = simulation.generate_frame_data(frame)
        frames_data.append(data)
        if (frame + 1) % 30 == 0:
            print(f"  ✓ {frame + 1}/{simulation.total_frames} 프레임")
    
    Path('models/animation_frames').mkdir(parents=True, exist_ok=True)
    output_path = 'models/animation_frames/relay_failure_frames.json'
    with open(output_path, 'w') as f:
        json.dump(frames_data, f)
    print(f"  ✓ 프레임 데이터: {output_path}")
    return frames_data

def create_matplotlib_animation(simulation, frames_data):
    print("\n[2/3] Matplotlib 애니메이션 생성 중...")
    fig, ax = plt.subplots(figsize=(14, 12), dpi=100)
    plt.rcParams['font.sans-serif'] = ['AppleGothic', 'DejaVu Sans']
    
    colors = {'healthy': '#00AA00', 'degraded': '#FFAA00', 'disconnected': '#FF0000'}
    link_widths = {'healthy': 2, 'degraded': 1.5, 'disconnected': 1}
    
    def setup_background():
        ax.clear()
        grid_background = np.where(simulation.grid == 1, 0.3, 1.0)
        ax.imshow(grid_background, cmap='gray', origin='lower',
                 extent=[0, simulation.grid_size * simulation.resolution_m,
                        0, simulation.grid_size * simulation.resolution_m],
                 alpha=0.5)
        ax.set_xlim(0, 5000)
        ax.set_ylim(0, 5000)
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
    
    def update_frame(frame_idx):
        setup_background()
        frame_data = frames_data[frame_idx]
        time = frame_data['time']
        
        for link in frame_data['links']:
            uav1 = frame_data['uavs'][link['from'] - 1]
            uav2 = frame_data['uavs'][link['to'] - 1]
            status = link['status']
            linestyle = '--' if status == 'disconnected' else '-'
            ax.plot([uav1['x'], uav2['x']], [uav1['y'], uav2['y']],
                   color=colors[status], linewidth=link_widths[status],
                   alpha=0.7, linestyle=linestyle)
        
        for uav in frame_data['uavs']:
            status = uav['status']
            color = colors[status]
            circle = Circle((uav['x'], uav['y']), 15, color=color,
                          ec='black', linewidth=2, alpha=0.8)
            ax.add_patch(circle)
            if status == 'healthy':
                comm_circle = Circle((uav['x'], uav['y']), 150,
                                   fill=False, edgecolor=color, linewidth=1,
                                   alpha=0.2, linestyle=':')
                ax.add_patch(comm_circle)
            ax.text(uav['x'], uav['y'], f"UAV{uav['id']}", fontsize=10,
                   fontweight='bold', ha='center', va='center', color='white')
        
        ax.text(0.02, 0.98, f"Time: {time:.2f}s", transform=ax.transAxes,
               fontsize=14, fontweight='bold', verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        ax.set_title('UAV Network - Relay Failure\n(Samsung Station)', fontsize=14, fontweight='bold')
    
    anim = FuncAnimation(fig, update_frame, frames=len(frames_data),
                        interval=1000/simulation.fps, repeat=True, blit=False)
    
    output_path = 'models/uav_network_animation.gif'
    Path('models').mkdir(parents=True, exist_ok=True)
    print(f"  ✓ GIF 저장 중...")
    writer = PillowWriter(fps=simulation.fps)
    anim.save(output_path, writer=writer)
    print(f"  ✓ GIF: {output_path}")
    plt.close()

def create_plotly_animation(simulation, frames_data):
    print("\n[3/3] HTML 인터랙티브 애니메이션 생성 중...")
    try:
        import plotly.graph_objects as go
    except:
        print("  ⚠️  Plotly 설치 필요")
        return
    
    frames = []
    colors = {'healthy': '#00AA00', 'degraded': '#FFAA00', 'disconnected': '#FF0000'}
    
    for frame_data in frames_data:
        uav_x = [uav['x'] for uav in frame_data['uavs']]
        uav_y = [uav['y'] for uav in frame_data['uavs']]
        uav_text = [f"UAV{uav['id']}<br>{uav['status']}" for uav in frame_data['uavs']]
        uav_colors = [colors[uav['status']] for uav in frame_data['uavs']]
        
        link_x, link_y = [], []
        for link in frame_data['links']:
            uav1 = frame_data['uavs'][link['from'] - 1]
            uav2 = frame_data['uavs'][link['to'] - 1]
            link_x.extend([uav1['x'], uav2['x'], None])
            link_y.extend([uav1['y'], uav2['y'], None])
        
        frame = go.Frame(
            data=[
                go.Scatter(x=link_x, y=link_y, mode='lines', line=dict(color='gray', width=2)),
                go.Scatter(x=uav_x, y=uav_y, mode='markers+text',
                          marker=dict(size=20, color=uav_colors, line=dict(color='black', width=2)),
                          text=[f"UAV{i+1}" for i in range(len(uav_x))],
                          textposition='middle center',
                          hovertext=uav_text, hoverinfo='text'),
            ],
            name=f'{frame_data["time"]:.2f}s'
        )
        frames.append(frame)
    
    frame_data = frames_data[0]
    uav_x = [uav['x'] for uav in frame_data['uavs']]
    uav_y = [uav['y'] for uav in frame_data['uavs']]
    uav_colors = [colors[uav['status']] for uav in frame_data['uavs']]
    
    fig = go.Figure(
        data=[
            go.Scatter(x=[], y=[], mode='lines'),
            go.Scatter(x=uav_x, y=uav_y, mode='markers+text',
                      marker=dict(size=20, color=uav_colors, line=dict(color='black', width=2)),
                      text=[f"UAV{i+1}" for i in range(len(uav_x))],
                      textposition='middle center'),
        ],
        frames=frames,
    )
    
    fig.update_layout(
        title='UAV Network - Relay Failure (Samsung Station)',
        xaxis=dict(range=[0, 5000], title='X (m)', gridcolor='lightgray'),
        yaxis=dict(range=[0, 5000], title='Y (m)', gridcolor='lightgray'),
        width=1000, height=1000, plot_bgcolor='rgba(240,240,240,0.9)',
        updatemenus=[dict(type='buttons', showactive=False, y=0.95, x=0.05,
            buttons=[
                dict(label='▶ Play', method='animate',
                     args=[None, {'frame': {'duration': int(1000/30), 'redraw': True}, 'fromcurrent': True}]),
                dict(label='⏸ Pause', method='animate',
                     args=[[None], {'frame': {'duration': 0, 'redraw': False}}])
            ])]
    )
    
    sliders = [{'active': 0, 'steps': [
        {'args': [[f.name], {'frame': {'duration': 0, 'redraw': True}, 'mode': 'immediate'}],
         'method': 'animate', 'label': f.name}
        for f in frames
    ]}]
    fig.update_layout(sliders=sliders)
    
    output_path = 'models/uav_network_animation.html'
    fig.write_html(output_path)
    print(f"  ✓ HTML: {output_path}")

def main():
    print("=" * 70)
    print("UAV 네트워크 애니메이션 생성")
    print("=" * 70)
    
    simulation = UAVNetworkSimulation()
    print(f"\n설정: {simulation.num_uavs} 드론, {simulation.total_duration}초, {simulation.fps}fps")
    
    frames_data = create_animation_frames(simulation)
    create_matplotlib_animation(simulation, frames_data)
    create_plotly_animation(simulation, frames_data)
    
    print("\n" + "=" * 70)
    print("✓ 완료!")
    print("  - GIF: open models/uav_network_animation.gif")
    print("  - HTML: open models/uav_network_animation.html")
    print("=" * 70)

if __name__ == "__main__":
    main()
