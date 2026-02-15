import sklearn
from sklearn.linear_model import SGDClassifier
import numpy as np

print(f"Scikit-learn version: {sklearn.__version__}")

try:
    print("Testing SGDClassifier(class_weight='balanced')...")
    model = SGDClassifier(class_weight="balanced", loss="log_loss")
    print("Success: class_weight='balanced' accepted in init.")
except Exception as e:
    print(f"FAILED init: {e}")

try:
    print("Testing partial_fit with class_weight='balanced'...")
    X = np.random.rand(10, 5)
    y = np.random.randint(0, 2, 10)
    model.partial_fit(X, y, classes=[0, 1])
    print("Success: partial_fit accepted.")
except Exception as e:
    print(f"FAILED partial_fit: {e}")
