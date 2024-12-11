import numpy as np
import matplotlib.pyplot as plt

# Veri noktaları
data = np.array([
    [2, 10], [5, 8],[2, 5], [8, 4],  [7, 5],
    [6, 4], [1, 2], [4, 9]
])

# K-Means parametreleri
k = 2  # Küme sayısı
max_iterations = 3  # Maksimum iterasyon
np.random.seed(0)

# Rastgele küme merkezleri seçimi
centroids = data[:k]
def distance(p1,p2):
    return np.sqrt(np.sum(np.square(p1-p2)))
# K-Means algoritması
for iteration in range(max_iterations):
    # Her noktanın en yakın merkeze atanması
    labels = []
    for point in data:
        distances = []
        for centroid in centroids:
            distances.append(distance(point,centroid))
        labels.append(np.argmin(distances))  # En yakın merkezin indeksini al

    labels = np.array(labels)


    new_centroids = np.array([data[labels == i].mean(axis=0) for i in range(k)])

    # Eğer merkezler değişmiyorsa algoritmayı durdur
    if np.all(centroids == new_centroids):
        break
    centroids = new_centroids

# Sonuçları görselleştirme
colors = ['blue', 'green','yellow']
for i in range(k):
    cluster_points = data[labels == i]
    plt.scatter(cluster_points[:, 0], cluster_points[:, 1], c=colors[i], label=f'Cluster {i+1}')
    plt.text(centroids[i, 0], centroids[i, 1],
             f"({centroids[i, 0]:.2f}, {centroids[i, 1]:.2f})",
             fontsize=10, color='red')
plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='X', s=200, label='Centroids')

plt.title('K-Means Clustering (Manual Implementation, k=2)')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.show()
