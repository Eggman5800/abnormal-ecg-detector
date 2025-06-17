import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model

from data_simulation import generate_clean_ecg, anomaly_inject
from preprocessing import sliding_window, normalize_windows

# Step 1: Load trained model
#model = load_model("lstm_autoencoder.h5")
model = load_model("lstm_autoencoder.keras")

# Step 2: Generate test signal with anomalies
ecg_clean = generate_clean_ecg(2000)
ecg_anomalous, anomaly_indices = anomaly_inject(ecg_clean, num_anomalies=8)

# Step 3: Preprocess the test signal
windows = sliding_window(ecg_anomalous, window_size=64, step=16)
windows_norm = normalize_windows(windows)
X_test = windows_norm[..., np.newaxis]  # shape: (num_windows, 64, 1)

# Step 4: Predict and calculate reconstruction errors
reconstructed = model.predict(X_test, verbose=0)
#mse = np.mean(np.square(X_test - reconstructed), axis=(1, 2))  # per window (mean squared error)
mae = np.mean(np.abs(X_test - reconstructed), axis=(1, 2)) # per window (mean absolute error)


# Step 5: Detect anomalies by threshold
threshold = np.percentile(mae, 95)  # simple rule: top 5% are anomalies
anomalous_windows = np.where(mae > threshold)[0]

# Map window indices to ECG signal indices
window_size = 64
step = 16
anomaly_points = [i * step for i in anomalous_windows]

# Step 6: Plotting
plt.figure(figsize=(12, 4))
plt.plot(ecg_anomalous, label='ECG with anomalies')
plt.scatter(anomaly_indices, ecg_anomalous[anomaly_indices], color='blue', label='Injected Anomalies')
plt.scatter(anomaly_points, ecg_anomalous[anomaly_points], color='red', marker='x', label='Detected Anomalies')
plt.title("Anomaly Detection in ECG Signal")
plt.xlabel("Time")
plt.ylabel("Amplitude")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
