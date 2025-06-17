import numpy as np
from data_simulation import generate_clean_ecg, anomaly_inject
import matplotlib.pyplot as plt


def sliding_window(signal, window_size=64, step=16):
    windows = []
    for i in range(0, len(signal) - window_size + 1, step):
        window = signal[i:i + window_size]
        windows.append(window)
    return np.array(windows)

def normalize_windows(windows):
    normalized = (windows - np.mean(windows, axis=1, keepdims=True)) / \
                 (np.std(windows, axis=1, keepdims=True) + 1e-8)
    return normalized

def global_normalize(signal):
    return (signal - np.mean(signal)) / np.std(signal)


#Code for testing these functions

ecg_clean = generate_clean_ecg(2000)
ecg_with_anomaly, _ = anomaly_inject(ecg_clean)

windows = sliding_window(ecg_with_anomaly)
windows_normalized = normalize_windows(windows)

plt.plot(windows[50], label='Raw window')
plt.plot(windows_normalized[50], label='Normalized', linestyle='dashed')
plt.title("A Sample Window: Raw vs Normalized")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()