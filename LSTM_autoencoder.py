import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import LSTM, RepeatVector, TimeDistributed, Dense, Dropout
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
import joblib

from data_simulation import generate_clean_ecg
from preprocessing import sliding_window, normalize_windows

# Step 1: Prepare clean data
ecg_clean = generate_clean_ecg(2000)
windows = sliding_window(ecg_clean, window_size=64, step=16)
windows_normalized = normalize_windows(windows)
X = windows_normalized[..., np.newaxis]  # shape: (N, 64, 1)

# Train-validation split
X_train, X_val = train_test_split(X, test_size=0.2, random_state=42)

# Step 2: Define LSTM Autoencoder
model = Sequential([
    LSTM(32, activation='tanh', input_shape=(64, 1), return_sequences=False),
    RepeatVector(64),
    LSTM(32, activation='tanh', return_sequences=True),
    Dropout(0.2),
    TimeDistributed(Dense(1))
])

model.compile(optimizer='adam', loss='mae')  # Use MAE for better sensitivity

# Step 3: Train with EarlyStopping
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

history = model.fit(X_train, X_train,
                    validation_data=(X_val, X_val),
                    epochs=500,
                    batch_size=32,
                    callbacks=[early_stop],
                    verbose=1)

# Save best model (auto-overwrites same file)
model.save("lstm_autoencoder.keras")

# Plot training & validation loss values
plt.figure(figsize=(10, 5))
plt.plot(history.history['loss'], label='Training Loss', linewidth=2)
plt.plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
plt.title('LSTM Autoencoder Training Progress')
plt.xlabel('Epoch')
plt.ylabel('Loss (MAE)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Optionally save training history
joblib.dump(history.history, "training_history.pkl")

print("Model training complete and saved as lstm_autoencoder.keras")
