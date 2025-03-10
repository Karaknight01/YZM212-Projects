import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Dosyayı oku
file_path = "video_game_sales.xlsx"
df = pd.read_excel(file_path)

# Eksik verileri temizle
df = df.dropna()

# Hedef değişkeni oluştur (Global Sales üzerinden sınıflandırma)
df["Sales_Class"] = pd.qcut(df["Global_Sales (m)"], q=3, labels=[0, 1, 2])

# Sınıf dağılımını hesapla
class_distribution = df["Sales_Class"].value_counts()
class_percentages = class_distribution / class_distribution.sum() * 100

# Sınıf dağılımını yazdır
print("📊 Sınıf Dağılımı:")
print(class_distribution)
print("\n📊 Sınıf Dağılımı (Yüzde):")
print(class_percentages)

# Sınıf dağılımını görselleştir
plt.figure(figsize=(6, 4))
sns.barplot(x=class_distribution.index, y=class_distribution.values, hue=class_distribution.index, palette="viridis", legend=False)
plt.xlabel("Sınıf Etiketleri")
plt.ylabel("Örnek Sayısı")
plt.title("Sınıf Dağılımı")
plt.show()
