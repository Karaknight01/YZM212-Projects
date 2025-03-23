
import pandas as pd
import time
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# -------------------- Veri Yükleme --------------------
train_df = pd.read_csv("train_video_game_sales.csv")
test_df = pd.read_csv("test_video_game_sales.csv")

X_train = train_df.drop(columns=["Success"])
y_train = train_df["Success"]
X_test = test_df.drop(columns=["Success"])
y_test = test_df["Success"]

# -------------------- Model Eğitimi --------------------
print("="*60)
print("🔧 Logistic Regression (Scikit-learn) Training Started")
model = LogisticRegression(max_iter=1000)

start_train = time.time()
model.fit(X_train, y_train)
end_train = time.time()

print("✅ Training Completed")
print(f"🕒 Training Time: {end_train - start_train:.4f} seconds")
print("="*60)

# -------------------- Tahmin --------------------
print("\n🚀 Starting Prediction...")
start_test = time.time()
y_pred = model.predict(X_test)
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
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='BuPu', xticklabels=[0,1], yticklabels=[0,1])
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix - Scikit-learn Logistic Regression")
plt.tight_layout()
plt.savefig("confusion_matrix_scikit.png")
plt.show()
