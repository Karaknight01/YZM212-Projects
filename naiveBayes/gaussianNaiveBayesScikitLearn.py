from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
import time
import numpy as np
import os

# Dosyaların olduğu klasör
current_dir = os.path.dirname(__file__)  # Eğer hata alırsan elle dizini yaz

# NumPy dosyalarını yükle
X_train = np.load(os.path.join(current_dir, "X_train.npy"))
X_test = np.load(os.path.join(current_dir, "X_test.npy"))
y_train = np.load(os.path.join(current_dir, "y_train.npy"))
y_test = np.load(os.path.join(current_dir, "y_test.npy"))

# Modeli oluştur
gnb = GaussianNB()

# Eğitim başlama zamanı
start_time = time.time()
gnb.fit(X_train, y_train)
train_time = time.time() - start_time  # Eğitim süresi

# Test başlama zamanı
start_time = time.time()
y_pred = gnb.predict(X_test)
test_time = time.time() - start_time  # Test süresi

# Model başarımını ölç
accuracy = accuracy_score(y_test, y_pred)

# Sonuçları yazdır
print(f"Model Doğruluğu: {accuracy:.4f}")
print(f"Eğitim Süresi: {train_time:.6f} saniye")
print(f"Test Süresi: {test_time:.6f} saniye")
