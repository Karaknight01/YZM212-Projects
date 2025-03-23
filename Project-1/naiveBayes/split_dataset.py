import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# File Path
file_path = "video_game_sales.xlsx"
df = pd.read_excel(file_path)

# Eksik verileri temizle
df = df.dropna()

# Sayısal olmayan sütunları kontrol et
non_numeric_cols = df.select_dtypes(include=['object']).columns
print("Sayısal olmayan sütunlar:", list(non_numeric_cols))

# Gereksiz sütunları çıkar (model için kullanmayacağız)
df = df.drop(columns=["Name", "Platform", "Genre", "Publisher"])

# Hedef değişkeni oluştur (Global Sales üzerinden sınıflandırma)
df["Sales_Class"] = pd.qcut(df["Global_Sales (m)"], q=3, labels=[0, 1, 2])

# Özellikler ve hedef değişkeni ayır
X = df.drop(columns=["Global_Sales (m)", "Sales_Class"])
y = df["Sales_Class"]

# Pandas ile veriyi bölme
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# NumPy formatına dönüştür
X_train = X_train.to_numpy()
X_test = X_test.to_numpy()
y_train = y_train.to_numpy()
y_test = y_test.to_numpy()

# Eğitim ve test verilerini kaydet
np.save("X_train.npy", X_train)
np.save("X_test.npy", X_test)
np.save("y_train.npy", y_train)
np.save("y_test.npy", y_test)

print("Eğitim ve test verileri başarıyla kaydedildi (NumPy formatında)!")
