
import pandas as pd
import numpy as np
import time
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# -------------------- Yardımcı Fonksiyonlar --------------------

def sigmoid(z):
    z = np.clip(z, -500, 500)  # overflow önlenir
    return 1 / (1 + np.exp(-z))

def compute_cost(X, y, weights):
    m = len(y)
    h = sigmoid(np.dot(X, weights))
    epsilon = 1e-5
    cost = -(1/m) * (np.dot(y, np.log(h + epsilon)) + np.dot((1 - y), np.log(1 - h + epsilon)))
    return cost

def gradient_descent(X, y, weights, learning_rate, iterations):
    m = len(y)
    cost_history = []
    for i in range(iterations):
        h = sigmoid(np.dot(X, weights))
        gradient = np.dot(X.T, (h - y)) / m
        weights -= learning_rate * gradient
        cost = compute_cost(X, y, weights)
        cost_history.append(cost)
    return weights, cost_history

# -------------------- Veri Yükleme --------------------

train_df = pd.read_csv("train_video_game_sales.csv")
test_df = pd.read_csv("test_video_game_sales.csv")

X_train = train_df.drop(columns=["Success"]).values
y_train = train_df["Success"].values
X_test = test_df.drop(columns=["Success"]).values
y_test = test_df["Success"].values

# Bias sütunu ekle
X_train = np.hstack((np.ones((X_train.shape[0], 1)), X_train))
X_test = np.hstack((np.ones((X_test.shape[0], 1)), X_test))

weights = np.zeros(X_train.shape[1])

# -------------------- Model Eğitimi --------------------

print("="*60)
print("🔧 Logistic Regression (Custom Implementation) Training Started")
start_train = time.time()
weights, cost_history = gradient_descent(X_train, y_train, weights, learning_rate=0.01, iterations=1000)
end_train = time.time()
print("✅ Training Completed")
print(f"🕒 Training Time: {end_train - start_train:.4f} seconds")
print("="*60)

# -------------------- Tahmin --------------------

print("\n🚀 Starting Prediction...")
start_test = time.time()
y_pred_probs = sigmoid(np.dot(X_test, weights))
y_pred = (y_pred_probs >= 0.5).astype(int)
end_test = time.time()
print("✅ Prediction Completed")
print(f"🕒 Prediction Time: {end_test - start_test:.4f} seconds")
print("="*60)

# -------------------- Değerlendirme --------------------

conf_matrix = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred)

print("\n📊 Classification Report:")
print(report)

print("\n📉 Confusion Matrix:")
print(conf_matrix)

# -------------------- Görselleştirme --------------------

plt.figure(figsize=(6,4))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='YlGnBu', xticklabels=[0,1], yticklabels=[0,1])
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix - Custom Logistic Regression")
plt.tight_layout()
plt.savefig("confusion_matrix_custom.png")
plt.show()
