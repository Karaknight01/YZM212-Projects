# YZM212 - Logistic Regression ile Sınıflandırma

Bu proje, Logistic Regression algoritmasını hem Scikit-learn kütüphanesi hem de özel bir Python implementasyonu ile gerçekleştirerek performanslarını karşılaştırmaktadır.

## 1. Problem Tanımı

Video oyunu satışlarına dair bir veri seti kullanarak, oyunların başarılı olup olmadığını sınıflandıran bir model geliştirilmektedir.  
Başarı ölçütü olarak `Global_Sales (m)` değişkeninin ortalaması alınarak oyunlar "başarılı (1)" ve "başarısız (0)" olarak etiketlenmiştir.

## 2. Veri

Projede `video_game_sales.xlsx` veri seti kullanılmıştır.  
Veri seti:

- 1000 örnekten oluşmaktadır
- Kullanılan sayısal özellikler: `Rank`, `Year`, `NA_Sales`, `EU_Sales`, `JP_Sales`, `Other_Sales`, `Global_Sales`
- Eksik veriler temizlenmiş, sadece sayısal sütunlar seçilmiştir
- `Success` sütunu hedef değişken olarak oluşturulmuştur

Veri eğitim (%80) ve test (%20) seti olarak ayrılmıştır.

## 3. Yöntem

Projede aşağıdaki adımlar izlenmiştir:

- Verinin analiz edilmesi ve sınıf etiketlerinin oluşturulması (`split_dataset.py`)
- Scikit-learn Logistic Regression modelinin eğitilmesi (`LogisticRegressionScikitLearn.py`)
- Özel Logistic Regression algoritmasının uygulanması (`logisticRegressionBayes.py`)
- Karmaşıklık matrisi ve sınıflandırma metrikleri ile modellerin karşılaştırılması

## 4. Sonuçlar

Performans karşılaştırmasında aşağıdaki metrikler kullanılmıştır:

- Doğruluk Oranı (Accuracy)
- Eğitim ve Test Süreleri
- Karmaşıklık Matrisi (Confusion Matrix)
- Precision, Recall, F1-Score

| Model | Doğruluk (Accuracy) | Eğitim Süresi | Test Süresi |
|-------|---------------------|----------------|-------------|
| Scikit-learn | `YAZILACAK` | `YAZILACAK` | `YAZILACAK` |
| Özel Model | `YAZILACAK` | `YAZILACAK` | `YAZILACAK` |

Confusion matrix görselleri proje dizinine `.png` olarak kaydedilmektedir.

## 5. Tartışma ve Değerlendirme

Sınıf dağılımı dengeli olduğundan doğruluk oranı uygun bir değerlendirme metriğidir.  
Bunun yanında Precision, Recall ve F1-score metrikleri sınıflandırma başarısının detaylarını daha net ortaya koymaktadır.

Scikit-learn modeli optimize edilmiş bir yapıya sahip olduğundan daha hızlıdır.  
Ancak özel implementasyon, algoritmanın temel prensiplerini daha iyi anlamak açısından eğiticidir.

Karmaşıklık matrisi, modelin hangi sınıflarda hata yaptığını gözlemleme açısından oldukça faydalıdır.

## 6. Kurulum ve Çalıştırma

Projeyi çalıştırmak için:

Gerekli kütüphaneleri yüklemek için:

```
pip install -r requirements.txt
```

Veri setini eğitim/test olarak ayırmak için:

```
python split_dataset.py
```

Modelleri çalıştırmak için:

```
python LogisticRegressionScikitLearn.py
python logisticRegressionBayes.py
```