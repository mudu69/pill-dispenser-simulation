import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import iirnotch, lfilter
from pykalman import KalmanFilter

# First I try to simulate load cell data
np.random.seed(42)
time = np.linspace(0, 10, 1000)
true_weight = np.zeros_like(time)
true_weight[200:250] = 1.0  # Simulated pill dispensing event
true_weight[600:650] = 1.0  # another event
noise = 0.05 * np.random.randn(len(time))
powerline_noise = 0.1 * np.sin(2 * np.pi * 60 * time)
raw_signal = true_weight + noise + powerline_noise

# Moving average filter - first filter
def moving_average(signal, window_size=10):
    return np.convolve(signal, np.ones(window_size)/window_size, mode='same')

# Notch filter at 60Hz - second filter applied
def apply_notch_filter(signal, fs=1000, f0=60.0, Q=30.0):
    b, a = iirnotch(f0, Q, fs)
    return lfilter(b, a, signal)

# Kalman filter - third filter applied
def apply_kalman_filter(signal):
    kf = KalmanFilter(initial_state_mean=0, n_dim_obs=1)
    kf = kf.em(signal, n_iter=5)
    (filtered_state_means, _) = kf.filter(signal)
    return filtered_state_means.flatten()

# Applying filters step by step
ma_signal = moving_average(raw_signal)
notch_signal = apply_notch_filter(ma_signal)
kalman_signal = apply_kalman_filter(notch_signal)

# Simple threshold-based pill detection
def detect_dispense_events(filtered_signal, threshold=0.5, min_duration=10):
    above_threshold = filtered_signal > threshold
    events = []
    start = None
    for i, val in enumerate(above_threshold):
        if val and start is None:
            start = i
        elif not val and start is not None:
            if i - start >= min_duration:
                events.append((start, i))
            start = None
    return events

detected_events = detect_dispense_events(kalman_signal)

# Plotting
plt.figure(figsize=(15, 8))
plt.plot(time, raw_signal, label='Raw Load Cell Signal', alpha=0.4)
plt.plot(time, kalman_signal, label='Filtered Signal (MA + Notch + Kalman)', linewidth=2)

for start, end in detected_events:
    plt.axvspan(time[start], time[end], color='green', alpha=0.3, label='Detected Event' if start == detected_events[0][0] else "")

plt.xlabel('Time (s)')
plt.ylabel('Signal')
plt.title('Load Cell Signal Processing and Pill Dispense Detection')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

detected_events_time = [(time[start], time[end]) for start, end in detected_events]
detected_events_time
