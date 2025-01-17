import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import MeanShift

# Adım 1: Rastgele 8 adet (x, y) koordinatlı nokta oluşturma
np.random.seed(42)  # Rastgelelikte tekrarlanabilirlik için
X = np.random.rand(30, 2) * 10  # 0-10 aralığında rastgele noktalar

# Adım 2: Oluşturulan veri noktalarını görselleştirme
plt.figure(figsize=(10, 7))
plt.scatter(X[:, 0], X[:, 1], marker='o')
plt.title('Oluşturulan Rastgele Veri Noktaları')
plt.xlabel('X')
plt.ylabel('Y')
plt.show()

# Adım 3: Means-Shift modelini oluşturma
ms = MeanShift()

# Adım 4: Modeli oluşturulan veri noktalarıyla eğitme
ms.fit(X)

# Adım 5: Küme merkezlerini ve etiketleri alma
cluster_centers = ms.cluster_centers_  # Küme merkezleri
labels = ms.labels_  # Her noktanın ait olduğu küme etiketleri

# Adım 6: Kümelerin sayısını belirleme
n_clusters_ = len(np.unique(labels))

# Adım 7: Küme bilgilerini yazdırma
print(f'Kümelerin sayısı: {n_clusters_}')
print(f'Küme merkezleri: {cluster_centers}')

# Adım 8: Sonuçları görselleştirme
plt.figure(figsize=(10, 7))
# Veri noktalarını küme etiketlerine göre renkli olarak gösterme
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='rainbow', marker='o')
# Küme merkezlerini siyah çarpı işaretiyle gösterme
plt.scatter(cluster_centers[:, 0], cluster_centers[:, 1], marker='x', color='black', s=200)
plt.title('Means-Shift Kümeleme (Rastgele 8 Nokta)')
plt.xlabel('X')
plt.ylabel('Y')
plt.show()
