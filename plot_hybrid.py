import json
import math
import matplotlib.pyplot as plt
import numpy as np
import os

base_dir = '/mnt/d/ClaudeWorkspace/Code/badminton_audio_mvp'

def load_vision_data(path):
    with open(path, 'r') as f:
        return json.load(f)

def compute_vision_velocities(vision_data):
    velocities = []
    for i in range(1, len(vision_data)):
        prev = vision_data[i-1]
        curr = vision_data[i]
        dt = curr['time'] - prev['time']
        if dt <= 0: continue
        
        frame_motion = 0
        for cp in curr['persons']:
            min_dist = float('inf')
            for pp in prev['persons']:
                dist = math.hypot(cp['cx'] - pp['cx'], cp['cy'] - pp['cy'])
                if dist < min_dist:
                    min_dist = dist
            if min_dist != float('inf') and min_dist < 400:
                frame_motion += min_dist
                
        velocity = frame_motion / dt
        velocities.append((curr['time'], velocity))
    return velocities

vision_data = load_vision_data(os.path.join(base_dir, 'vision_metrics.json'))
raw_vels = compute_vision_velocities(vision_data)

# smooth slightly
smoothed = []
for i in range(len(raw_vels)):
    t_curr = raw_vels[i][0]
    window_vals = [v[1] for v in raw_vels if abs(v[0] - t_curr) <= 1.0/2]
    avg_vel = np.mean(window_vals) if window_vals else 0
    smoothed.append((t_curr, avg_vel))

times = [v[0] for v in smoothed]
vels = [v[1] for v in smoothed]

plt.figure(figsize=(15, 6))
plt.plot(times, vels, label='Smoothed Velocity (0.5s)')
plt.axhline(np.percentile(vels, 80), color='r', linestyle='--', label='80th Percentile')
plt.axhline(np.percentile(vels, 90), color='g', linestyle='--', label='90th Percentile')
plt.xlabel('Time (s)')
plt.ylabel('Velocity (px/s)')
plt.title('Player Movement Burst Over Time')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(base_dir, 'velocity_burst_plot.png'))
print("Saved velocity_burst_plot.png")
