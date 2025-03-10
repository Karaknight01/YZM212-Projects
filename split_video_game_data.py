import pandas as pd
from sklearn.model_selection import train_test_split

# GitHub'daki dosya URL'si
github_url = "https://raw.githubusercontent.com/kullaniciadi/repoadi/main/video_game_sales.xlsx"

# Veriyi oku
df = pd.read_excel(file_path)

# Eksik verileri temizle
df = df.dropna()

# Özellikler (X) ve hedef değişken (y)
X = df[["Year", "NA_Sales (m)", "EU_Sales (m)", "JP_Sales (m)", "Other_Sales (m)"]]

# Eğer 'Sales_Class' sütunu yoksa, toplam satışlara göre sınıflandırma ekleyelim
df["Sales_Class"] = pd.qcut(df["Global_Sales (m)"], q=3, labels=[0, 1, 2])

y = df["Sales_Class"]

# Veriyi %80 eğitim, %20 test olarak böl
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Veri setlerini CSV olarak kaydet
X_train.to_csv("X_train.csv", index=False)
X_test.to_csv("X_test.csv", index=False)
y_train.to_csv("y_train.csv", index=False)
y_test.to_csv("y_test.csv", index=False)

print("Eğitim ve test verileri başarıyla kaydedildi!")
