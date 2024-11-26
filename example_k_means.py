import numpy as np
import cv2
import matplotlib.pyplot as plt

def initialize_centroids(data, k):
    """Veri içinden rastgele k merkez noktası seç"""
    indices = np.random.choice(data.shape[0], k, replace=False)
    return data[indices]

def assign_clusters(data, centroids):
    """Veriyi en yakın merkez noktalarına at"""
    distances = np.sqrt(((data - centroids[:, np.newaxis])**2).sum(axis=2))
    return np.argmin(distances, axis=0)

def update_centroids(data, labels, k):
    """Yeni merkez noktalarını hesapla"""
    new_centroids = np.array([data[labels == i].mean(axis=0) for i in range(k)])
    return new_centroids

def k_means(data, k, max_iters=100):
    centroids = initialize_centroids(data, k)
    for _ in range(max_iters):
        labels = assign_clusters(data, centroids)
        new_centroids = update_centroids(data, labels, k)
        if np.all(centroids == new_centroids):
            break
        centroids = new_centroids
    return centroids, labels

# Görüntüyü yükle
image = cv2.imread('image.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Görüntüyü piksellere ayır ve 2D diziye dönüştür
pixels = image.reshape(-1, 3).astype(np.float32)

# K-Means algoritmasını uygula
k = 4
centroids, labels = k_means(pixels, k)

# Segmentleri yeniden renklendir
segmented_img = centroids[labels].reshape(image.shape).astype(np.uint8)

# Görüntüyü göster
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title('Orijinal Görüntü')
plt.imshow(image)
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title('Segmented Görüntü')
plt.imshow(segmented_img)
plt.axis('off')

plt.show()
