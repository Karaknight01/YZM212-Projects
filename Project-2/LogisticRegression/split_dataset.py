
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Veri dosyasını oku
df = pd.read_excel("video_game_sales.xlsx")

# Global satış ortalamasına göre ikili sınıflandırma etiketi oluştur
threshold = df["Global_Sales (m)"].mean()
df["Success"] = (df["Global_Sales (m)"] >= threshold).astype(int)

# Sayısal özellikleri al, hedef değişkeni ayır
X = df.select_dtypes(include=[np.number]).drop(columns=["Success"])
y = df["Success"]

# %80 eğitim, %20 test bölmesi
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Eğitim ve test veri setlerini birleştir
train_df = X_train.copy()
train_df["Success"] = y_train

test_df = X_test.copy()
test_df["Success"] = y_test

# CSV olarak kaydet
train_df.to_csv("train_video_game_sales.csv", index=False)
test_df.to_csv("test_video_game_sales.csv", index=False)
