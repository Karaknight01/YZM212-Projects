import numpy as np
import time
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix

# Custom Gaussian Naive Bayes Sınıfı
class CustomGaussianNB:
    def fit(self, X, y):
        self.classes = np.unique(y)
        self.means = {}
        self.variances = {}
        self.priors = {}

        for c in self.classes:
            X_c = X[y == c]
            self.means[c] = np.mean(X_c, axis=0)
            self.variances[c] = np.var(X_c, axis=0) + 1e-6  # Sıfır varyans önleme
            self.priors[c] = X_c.shape[0] / X.shape[0]

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


# NumPy veri setlerini yükle
X_train = np.load("X_train.npy")
X_test = np.load("X_test.npy")
y_train = np.load("y_train.npy")
y_test = np.load("y_test.npy")

# --- Scikit-learn Gaussian Naive Bayes ---
gnb = GaussianNB()

start_time = time.time()
gnb.fit(X_train, y_train)
sklearn_train_time = time.time() - start_time

start_time = time.time()
y_pred_sklearn = gnb.predict(X_test)
sklearn_test_time = time.time() - start_time

sklearn_accuracy = accuracy_score(y_test, y_pred_sklearn)

# --- Custom Gaussian Naive Bayes ---
custom_nb = CustomGaussianNB()

start_time = time.time()
custom_nb.fit(X_train, y_train)
custom_train_time = time.time() - start_time

start_time = time.time()
y_pred_custom = custom_nb.predict(X_test)
custom_test_time = time.time() - start_time

custom_accuracy = accuracy_score(y_test, y_pred_custom)

# --- Çıktıları yazdır ---
print("Scikit-learn Gaussian Naive Bayes")
print(f"Doğruluk: {sklearn_accuracy:.4f}")
print(f"Eğitim Süresi: {sklearn_train_time:.6f} saniye")
print(f"Test Süresi: {sklearn_test_time:.6f} saniye\n")

print("Custom Gaussian Naive Bayes")
print(f"Doğruluk: {custom_accuracy:.4f}")
print(f"Eğitim Süresi: {custom_train_time:.6f} saniye")
print(f"Test Süresi: {custom_test_time:.6f} saniye\n")

# --- Grafiklerle Karşılaştırma ---
plt.figure(figsize=(14, 5))

# Doğruluk Karşılaştırması
plt.subplot(1, 3, 1)
sns.barplot(x=["Scikit-learn", "Custom"], y=[sklearn_accuracy, custom_accuracy], hue=["Scikit-learn", "Custom"], legend=False, palette="viridis")
plt.ylabel("Doğruluk")
plt.title("Model Doğruluk Karşılaştırması")

# Süre Karşılaştırması
labels = ["Sklearn Fit", "Sklearn Predict", "Custom Fit", "Custom Predict"]
times = [sklearn_train_time, sklearn_test_time, custom_train_time, custom_test_time]

plt.subplot(1, 3, 2)
sns.barplot(x=labels, y=times, hue=labels, legend=False, palette="magma")
plt.ylabel("Süre (saniye)")
plt.title("Eğitim ve Test Süreleri")

# Confusion Matrix
plt.subplot(1, 3, 3)
cm = confusion_matrix(y_test, y_pred_sklearn)  # Sadece sklearn modeli için ekleniyor
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["Class 0", "Class 1", "Class 2"], yticklabels=["Class 0", "Class 1", "Class 2"])
plt.xlabel("Tahmin Edilen")
plt.ylabel("Gerçek Değer")
plt.title("Confusion Matrix - Sklearn")

plt.tight_layout()
plt.show()
