import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

# 1. Load Dataset
df = pd.read_csv("Cloud_Dataset.csv")


df["target"] = (
    (df["utilization"] < 30) &
    (df["cpu_usage"] < 50) &
    (df["memory_usage"] < 60)
).astype(int)

noise = np.random.rand(len(df)) < 0.1
df.loc[noise, "target"] = 1 - df.loc[noise, "target"]

# 3. Select Features
features = ["cpu_usage","memory_usage","disk_io","latency_ms","throughput","cost"]

X = df[features]
y = df["target"]

# 4. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 5. Train Model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 6. Predictions
pred = model.predict(X_test)

# 7. Evaluation Metrics
accuracy = accuracy_score(y_test, pred)
precision = precision_score(y_test, pred)
recall = recall_score(y_test, pred)
f1 = f1_score(y_test, pred)
cm = confusion_matrix(y_test, pred)

# 8. Cross Validation (NEW 🔥)
cv_scores = cross_val_score(model, X, y, cv=5)

# 9. Print Results
print("\n===== MODEL PERFORMANCE =====")
print(f"Accuracy        : {accuracy:.2f}")
print(f"Precision       : {precision:.2f}")
print(f"Recall          : {recall:.2f}")
print(f"F1 Score        : {f1:.2f}")
print(f"Confusion Matrix:\n{cm}")
print(f"Cross-val Score : {cv_scores.mean():.2f}")

# 10. Save Model
joblib.dump(model, "cloud_model.pkl")
print("\nModel saved as cloud_model.pkl")

# 11. Save Metrics
with open("model_metrics.txt", "w") as f:
    f.write("MODEL PERFORMANCE\n")
    f.write("=================\n")
    f.write(f"Accuracy: {accuracy}\n")
    f.write(f"Precision: {precision}\n")
    f.write(f"Recall: {recall}\n")
    f.write(f"F1 Score: {f1}\n")
    f.write(f"Confusion Matrix:\n{cm}\n")
    f.write(f"Cross Validation: {cv_scores.mean()}\n")

print("Metrics saved as model_metrics.txt")

# 12. Feature Importance
importances = model.feature_importances_

plt.figure()
plt.barh(features, importances)
plt.xlabel("Importance")
plt.title("Feature Importance")
plt.show()