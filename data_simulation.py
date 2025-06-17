import numpy as np
import matplotlib.pyplot as plt

def generate_clean_ecg(length = 1000,noise_level = 0.01):
    t = np.linspace(0, 1, length)
    ecg = np.sin(2 * np.pi * 5 * t)  # base heart rhythm
    ecg += 0.5 * np.sin(2 * np.pi * 15 * t)  # mimic P, QRS, T complexity
    ecg += noise_level * np.random.randn(length)
    return ecg

def anomaly_inject(ecg_signal, num_anomalies=5, severity=2.5):
    ecg = ecg_signal.copy()
    length = len(ecg)
    anomaly_indices = np.random.randint(100, length-100, size=num_anomalies)
    for idx in anomaly_indices:
        ecg[idx:idx+10] += severity * (np.random.rand(10) - 0.5)
    return ecg, anomaly_indices


#Code for testing these functions

ecg_clean = generate_clean_ecg(2000)
ecg_with_anomaly, anomaly_locs = anomaly_inject(ecg_clean, num_anomalies=8)

plt.figure(figsize=(12, 4))
plt.plot(ecg_with_anomaly, label='ECG with anomalies')
plt.scatter(anomaly_locs, ecg_with_anomaly[anomaly_locs], color='red', label='Anomaly injected', zorder=3)
plt.title("Simulated ECG with Injected Anomalies")
plt.xlabel("Time")
plt.ylabel("Amplitude")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
