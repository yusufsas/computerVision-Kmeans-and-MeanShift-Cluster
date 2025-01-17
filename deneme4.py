import numpy as np
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
import time

# Piksel güncelleme fonksiyonu
def update_pixel(idx, data, bandwidth):
    point = data[idx]
    distances = np.linalg.norm(data - point, axis=1)
    weights = np.exp(-distances**2 / (2 * bandwidth**2))
    numerator = np.sum(data * weights[:, None], axis=0)
    denominator = np.sum(weights)
    new_point = numerator / denominator
    return new_point

def mean_shift_manual(data, bandwidth=2, max_iter=100, tol=1e-3):
    start = time.time()
    flat_data = data.copy()

    for i in range(max_iter):
        old_data = flat_data.copy()

        # Paralel işlemle veri noktalarını güncelleme
        flat_data = np.array(Parallel(n_jobs=-1)(
            delayed(update_pixel)(idx, flat_data, bandwidth) for idx in range(len(flat_data))
        ))

        # Güncellemenin boyutunu kontrol et
        if np.linalg.norm(old_data - flat_data) < tol:
            break

    end = time.time()
    total_time = end - start

    return flat_data, total_time

# Adım 1: Rastgele veri noktaları oluşturma
np.random.seed(42)
X = np.random.rand(10, 2) * 10

# Adım 2: Mean-Shift'i manuel olarak uygulama
bandwidth = 5.0
max_iter = 5
tol = 1e-3
segmented_data, elapsed_time = mean_shift_manual(X, bandwidth=bandwidth, max_iter=max_iter, tol=tol)

# Adım 3: Küme merkezlerini belirleme
unique_points = np.unique(np.round(segmented_data, decimals=3), axis=0)

# Küme merkezlerini ve etiketleri belirleme
cluster_centers = unique_points
labels = np.zeros(len(X))
for i, center in enumerate(cluster_centers):
    labels[np.all(np.isclose(segmented_data, center, atol=tol), axis=1)] = i

# Adım 4: Sonuçları yazdırma
print(f'Manuel Mean-Shift süresi: {elapsed_time:.2f} saniye')
print(f'Kümelerin sayısı: {len(cluster_centers)}')
print(f'Küme merkezleri:\n{cluster_centers}')

# Adım 5: Sonuçları görselleştirme
plt.figure(figsize=(10, 7))
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='rainbow', marker='o')
plt.scatter(cluster_centers[:, 0], cluster_centers[:, 1], marker='x', color='black', s=200)
plt.title('Manuel Mean-Shift Kümeleme')
plt.xlabel('X')
plt.ylabel('Y')
plt.show()
