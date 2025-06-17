import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model

from data_simulation import generate_clean_ecg, anomaly_inject
from preprocessing import sliding_window, normalize_windows

# Step 1: Set seed for reproducibility
np.random.seed(42)

# Step 2: Load trained model
model = load_model("lstm_autoencoder.keras")

# Step 3: Generate test signal with anomalies (consistent)
ecg_clean = generate_clean_ecg(2000)
ecg_anomalous, anomaly_indices = anomaly_inject(ecg_clean, num_anomalies=8)

# Step 4: Preprocess the test signal
windows = sliding_window(ecg_anomalous, window_size=64, step=16)
windows_norm = normalize_windows(windows)
X_test = windows_norm[..., np.newaxis]  # shape: (num_windows, 64, 1)

# Step 5: Predict and calculate reconstruction errors
reconstructed = model.predict(X_test, verbose=0)
mae = np.mean(np.abs(X_test - reconstructed), axis=(1, 2))  # using MAE

# Step 6: Detect anomalies by threshold (95th percentile of MAE)
threshold = np.percentile(mae, 95)
anomalous_windows = np.where(mae > threshold)[0]

# Map window indices to ECG signal indices
window_size = 64
step = 16
anomaly_points = [i * step for i in anomalous_windows]

# Step 7: Plot injected anomalies for reference
plt.figure(figsize=(12, 3))
plt.plot(ecg_anomalous, label='Anomalous ECG')
plt.scatter(anomaly_indices, ecg_anomalous[anomaly_indices], color='blue', label='Injected Anomalies')
plt.title("Injected Anomalies in ECG")
plt.xlabel("Time")
plt.ylabel("Amplitude")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Step 8: Plot detected anomalies
plt.figure(figsize=(12, 4))
plt.plot(ecg_anomalous, label='ECG with anomalies')
plt.scatter(anomaly_indices, ecg_anomalous[anomaly_indices], color='blue', label='Injected Anomalies')
plt.scatter(anomaly_points, ecg_anomalous[anomaly_points], color='red', marker='x', label='Detected Anomalies')
plt.title("Detected vs Injected Anomalies")
plt.xlabel("Time")
plt.ylabel("Amplitude")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
