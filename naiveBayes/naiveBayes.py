import numpy as np
import time
import os

class CustomGaussianNB:
    def fit(self, X, y):
        self.classes = np.unique(y)  # Sınıfları belirle
        self.means = {}
        self.variances = {}
        self.priors = {}

        for c in self.classes:
            X_c = X[y == c]  # Her sınıf için verileri filtrele
            self.means[c] = np.mean(X_c, axis=0)
            self.variances[c] = np.var(X_c, axis=0) + 1e-6  # Küçük bir sabit ekleyerek sıfır varyanstan kaçın
            self.priors[c] = X_c.shape[0] / X.shape[0]  # Öncelik olasılığı

    def gaussian_pdf(self, x, mean, var):
        return (1 / np.sqrt(2 * np.pi * var)) * np.exp(-((x - mean) ** 2) / (2 * var))

    def predict(self, X):
        predictions = []
        for x in X:
            posteriors = {}
            for c in self.classes:
                prior = np.log(self.priors[c])
                likelihood = np.sum(np.log(self.gaussian_pdf(x, self.means[c], self.variances[c])))
                posteriors[c] = prior + likelihood
            predictions.append(max(posteriors, key=posteriors.get))
        return np.array(predictions)


# Dosya yolu
current_dir = os.path.dirname(__file__)  # Eğer hata verirse, elle dizini yaz

# .npy dosyalarını oku
X_train = np.load(os.path.join(current_dir, "X_train.npy"))
X_test = np.load(os.path.join(current_dir, "X_test.npy"))
y_train = np.load(os.path.join(current_dir, "y_train.npy"))
y_test = np.load(os.path.join(current_dir, "y_test.npy"))

# Özel Gaussian Naive Bayes modelini oluştur
custom_nb = CustomGaussianNB()

# Eğitim başlama zamanı
start_time = time.time()
custom_nb.fit(X_train, y_train)
train_time = time.time() - start_time  # Eğitim süresi

# Test başlama zamanı
start_time = time.time()
y_pred = custom_nb.predict(X_test)
test_time = time.time() - start_time  # Test süresi

# Sonuçları yazdır
print(f"Model Doğruluğu: {np.mean(y_pred == y_test):.4f}")
print(f"Eğitim Süresi: {train_time:.6f} saniye")
print(f"Test Süresi: {test_time:.6f} saniye")
