import os
import numpy as np
import librosa
from scipy.signal import find_peaks

base_dir = '/mnt/d/ClaudeWorkspace/Code/badminton_audio_mvp'
audio_path = os.path.join(base_dir, 'track.wav')

start_offset = 50
duration = 55
y, sr = librosa.load(audio_path, sr=None, offset=start_offset, duration=duration)
hop_length = 512
energy = np.array([sum(abs(y[i:i+hop_length]**2)) for i in range(0, len(y), hop_length)])
times = librosa.frames_to_time(np.arange(len(energy)), sr=sr, hop_length=hop_length)
energy = energy / np.max(energy)

peaks, _ = find_peaks(energy, height=0.015, distance=int(sr * 0.2 / hop_length))
audio_hits = times[peaks]
for p in audio_hits:
    print(f"Peak at: {p:.2f}s")
