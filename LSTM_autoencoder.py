import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import LSTM, RepeatVector, TimeDistributed, Dense
#from tensorflow.keras import Sequential
#from tensorflow.keras.layers import LSTM, RepeatVector, TimeDistributed, Dense
from sklearn.model_selection import train_test_split
import joblib

from data_simulation import generate_clean_ecg
from preprocessing import sliding_window, normalize_windows, global_normalize

# Step 1: Prepare clean data
ecg_clean = generate_clean_ecg(2000)
ecg_clean = global_normalize(ecg_clean)
windows = sliding_window(ecg_clean, window_size=64, step=16)
windows_normalized = normalize_windows(windows)

# Reshape for LSTM [samples, timesteps, features]
X = windows_normalized[..., np.newaxis]  # shape = (N, 64, 1)

# Train-test split (only for clean training)
X_train, X_val = train_test_split(X, test_size=0.2, random_state=42)

# Step 2: Define LSTM Autoencoder
model = Sequential([
    LSTM(64, activation='relu', input_shape=(64, 1), return_sequences=True),
    LSTM(32, activation='relu', return_sequences=False),
    RepeatVector(64),
    LSTM(32, activation='relu', return_sequences=True),
    LSTM(64, activation='relu', return_sequences=True),
    TimeDistributed(Dense(1))
])


model.compile(optimizer='adam', loss='mae')

# Step 3: Train
history = model.fit(X_train, X_train,
                    validation_data=(X_val, X_val),
                    epochs=500,
                    batch_size=32,
                    verbose=1)

# Save model
#model.save("lstm_autoencoder.h5")
model.save("lstm_autoencoder.keras")
# Optionally, save training loss for plotting
joblib.dump(history.history, "training_history.pkl")

print("Model training complete and saved as lstm_autoencoder.keras")
