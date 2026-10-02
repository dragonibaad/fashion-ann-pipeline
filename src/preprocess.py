import os
import numpy as np
from sklearn.model_selection import train_test_split

raw_dir=os.path.join(os.path.dirname(__file__), "..", "data", "raw")
preprocessed_dir=os.path.join(os.path.dirname(__file__), "..", "data", "processed")
os.makedirs(preprocessed_dir, exist_ok=True)

print(f"Now loading raw data from {raw_dir}")
X_train_full=np.load(os.path.join(raw_dir, 'X_train.npy'))
y_train_full=np.load(os.path.join(raw_dir, 'y_train.npy'))
X_test=np.load(os.path.join(raw_dir, 'X_test.npy'))
y_test=np.load(os.path.join(raw_dir, 'y_test.npy'))

print("Now normalizing the data")
X_train_full=X_train_full.astype('float32') / 255.0
X_test=X_test.astype('float32') / 255.0

X_train, X_val, y_train, y_val=train_test_split(X_train_full, y_train_full, test_size=0.2, random_state=42)
np.save(os.path.join(preprocessed_dir, 'X_train.npy'), X_train)
np.save(os.path.join(preprocessed_dir, 'y_train.npy'), y_train)
np.save(os.path.join(preprocessed_dir, 'X_val.npy'), X_val)
np.save(os.path.join(preprocessed_dir, 'y_val.npy'), y_val)
np.save(os.path.join(preprocessed_dir, 'X_test.npy'), X_test)
np.save(os.path.join(preprocessed_dir, 'y_test.npy'), y_test)