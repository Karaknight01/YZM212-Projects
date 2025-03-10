# YZM212-Project

Naive Bayes ile Sınıflandırma

Bu proje, Gaussian Naive Bayes algoritmasını hem Scikit-learn kütüphanesi hem de özel bir Python implementasyonu ile gerçekleştirerek performanslarını karşılaştırmaktadır.

1. Problem Tanımı

Video oyunu satışlarına dair bir veri seti kullanarak, oyunların başarılı olup olmadığını sınıflandıran bir model geliştirilmektedir. 
Gaussian Naive Bayes algoritması kullanılarak, oyunları satış miktarına göre belirli sınıflara ayırmak hedeflenmektedir.

2. Veri

Projede video_game_sales.xlsx veri seti kullanılmıştır. Veri seti:

Global_Sales (m) değişkeni temel alınarak düşük, orta ve yüksek satış olmak üzere 3 sınıfa ayrılmıştır.

Eksik veriler temizlenerek sınıf dağılımı analiz edilmiştir.

3. Yöntem

Projede aşağıdaki adımlar izlenmiştir:

Veri setinin temizlenmesi ve sınıflandırılması (class_distribution_analysis.py)

Verinin eğitim ve test setlerine bölünmesi (split_dataset.py)

Scikit-learn Gaussian Naive Bayes modelinin eğitilmesi (gaussianNaiveBayesScikitLearn.py)

Özel bir Gaussian Naive Bayes algoritmasının uygulanması (naiveBayes.py)

Modellerin doğruluk ve hız bakımından karşılaştırılması (compare_models.py)

4. Sonuçlar

Performans karşılaştırmasında aşağıdaki metrikler kullanılmıştır:

Doğruluk Oranı (Accuracy)

Eğitim ve Test Süreleri

Karmaşıklık Matrisi (Confusion Matrix)

Sonuçlar, Scikit-learn'in GaussianNB modelinin genellikle daha hızlı ve optimize olduğunu, ancak özel implementasyonun da benzer doğruluk oranları sunduğunu göstermiştir.

5. Tartışma ve Değerlendirme

Sınıf dağılımı, değerlendirme metriklerinin seçimi açısından önemlidir.

Dengeli bir veri setinde doğruluk iyi bir metrik olabilir, ancak dengesiz veri setlerinde precision, recall ve F1-score gibi metrikler daha sağlıklı sonuçlar sunar.

Sınıf dengesizliği durumunda, SMOTE veya diğer veri düzenleme yöntemleri kullanılabilir.

Karmaşıklık matrisi, modelin hangi sınıflarda hata yaptığını anlamaya yardımcı olur ve yanlış tahminlerin etkisini gözlemlemeyi sağlar.

Kurulum ve Çalıştırma

Projeyi çalıştırmak için aşağıdaki komutları kullanabilirsiniz:

Gerekli kütüphaneleri yüklemek için:

pip install -r requirements.txt

Modelleri çalıştırmak için:

python compare_models.py

Bu sayede Scikit-learn ve Custom Naive Bayes modelleri karşılaştırılacak ve çıktılar görüntülenecektir.
