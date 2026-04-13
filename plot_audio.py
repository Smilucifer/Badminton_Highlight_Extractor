import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import os
import json

base_dir = '/mnt/d/ClaudeWorkspace/Code/badminton_audio_mvp'
audio_path = os.path.join(base_dir, 'track.wav')
vision_path = os.path.join(base_dir, 'vision_metrics.json')

start_offset = 50
duration = 55
y, sr = librosa.load(audio_path, sr=None, offset=start_offset, duration=duration)
hop_length = 512
energy = np.array([sum(abs(y[i:i+hop_length]**2)) for i in range(0, len(y), hop_length)])
times = librosa.frames_to_time(np.arange(len(energy)), sr=sr, hop_length=hop_length)
energy = energy / np.max(energy)

plt.figure(figsize=(15, 6))
plt.plot(times, energy, label='Audio Energy')
plt.axhline(0.015, color='r', linestyle='--', label='Hit Threshold')
plt.xlabel('Time (s)')
plt.ylabel('Normalized Energy')
plt.title('Audio Energy in 55s Clip')

# mark valid rallies
with open(os.path.join(base_dir, 'high_value_rallies.json'), 'r') as f:
    rallies = json.load(f)
for r in rallies:
    plt.axvspan(r['start_time'], r['end_time'], color='green', alpha=0.3, label='Detected Rally' if r == rallies[0] else None)
    
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(base_dir, 'audio_energy_plot.png'))
print("Saved audio_energy_plot.png")
