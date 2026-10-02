import os
import json
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

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
X_test = np.load(os.path.join(data_dir, 'X_test.npy'))
y_test = np.load(os.path.join(data_dir, 'y_test.npy'))

models_dir = os.path.join(os.path.dirname(__file__), "..", "models")
model_path = os.path.join(models_dir, "mnist_model.h5")
model = tf.keras.models.load_model(model_path)

loss, accuracy = model.evaluate(X_test, y_test, verbose=0)

y_pred_probs = model.predict(X_test)
y_pred = np.argmax(y_pred_probs, axis=1)

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')

os.makedirs(models_dir, exist_ok=True)
cm_path = os.path.join(models_dir, "confusion_matrix.png")
plt.savefig(cm_path)
plt.close()

metrics = {
    "test_loss": float(loss),
    "test_accuracy": float(accuracy)
}

root_dir = os.path.join(os.path.dirname(__file__), "..")
metrics_path = os.path.join(root_dir, "metrics.json")
with open(metrics_path, 'w') as f:
    json.dump(metrics, f, indent=4)

print(f"Test Loss: {loss:.4f}")
print(f"Test Accuracy: {accuracy:.4f}")
print(f"Confusion matrix saved to {cm_path}")
print(f"Metrics saved to {metrics_path}")