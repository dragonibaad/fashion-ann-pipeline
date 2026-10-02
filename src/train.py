import os
import yaml
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Flatten, Dense, Dropout

params_path = os.path.join(os.path.dirname(__file__), "..", "params.yaml")
with open(params_path, "r") as f:
    params = yaml.safe_load(f)

train_params = params.get("train", {})
dense_units = train_params.get("dense_units", 128)
dropout_rate = train_params.get("dropout_rate", 0.2)
learning_rate = train_params.get("learning_rate", 0.001)
epochs = train_params.get("epochs", 10)
batch_size = train_params.get("batch_size", 32)

physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    try:
        for gpu in physical_devices:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"Found {len(physical_devices)} GPU(s). Memory growth enabled.")
    except RuntimeError as e:
        print(f"Error setting memory growth: {e}")
else:
    print("No GPU found. Running on CPU.")

data_dir = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
X_train = np.load(os.path.join(data_dir, 'X_train.npy'))
y_train = np.load(os.path.join(data_dir, 'y_train.npy'))
X_val = np.load(os.path.join(data_dir, 'X_val.npy'))
y_val = np.load(os.path.join(data_dir, 'y_val.npy'))

model = Sequential([Flatten(input_shape=(28, 28)), Dense(dense_units, activation='relu'), Dropout(dropout_rate), Dense(10, activation='softmax')])

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
print("Now training the model")

history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_val, y_val))

os.makedirs(os.path.join(os.path.dirname(__file__), "..", "models"), exist_ok=True)
model_path = os.path.join(os.path.dirname(__file__), "..", "models", "mnist_model.h5")
model.save(model_path)
print(f"Model saved to {model_path}")

history_df = pd.DataFrame(history.history)
history_path = os.path.join(os.path.dirname(__file__), "..", "models", "history.csv")
history_df.to_csv(history_path, index=False)