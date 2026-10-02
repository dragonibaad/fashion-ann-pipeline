import tensorflow as tf
import numpy as np
import pandas as pd
import os

raw_dir = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
os.makedirs(raw_dir, exist_ok=True)
(X_train, y_train), (X_test, y_test)=tf.keras.datasets.fashion_mnist.load_data()
np.save(os.path.join(raw_dir, 'X_train.npy'), X_train)
np.save(os.path.join(raw_dir, 'y_train.npy'), y_train)
np.save(os.path.join(raw_dir, 'X_test.npy'), X_test)
np.save(os.path.join(raw_dir, 'y_test.npy'), y_test)
print(f"Raw data saved in {raw_dir}")