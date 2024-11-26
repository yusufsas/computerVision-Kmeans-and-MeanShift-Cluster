import numpy as np
import cv2
import matplotlib.pyplot as plt

def gaussian_kernel(distance, bandwidth):
    """Gaussian kernel fonksiyonu"""
    return (1 / (bandwidth * np.sqrt(2 * np.pi))) * np.exp(-0.5 * (distance / bandwidth) ** 2)

def mean_shift(data, bandwidth, max_iters=300):
    centroids = np.copy(data)
    for _ in range(max_iters):
        for i in range(centroids.shape[0]):
            distances = np.sqrt(((data - centroids[i]) ** 2).sum(axis=1))
            weights = gaussian_kernel(distances, bandwidth)
            numerator = (weights[:, np.newaxis] * data).sum(axis=0)
            denominator = weights.sum()
            new_centroid = numerator / denominator
            if np.linalg.norm(new_centroid - centroids[i]) < 1e-3:
                continue
            centroids[i] = new_centroid
    return np.unique(centroids, axis=0)

# Görüntüyü yükle
image = cv2.imread('image.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Görüntüyü piksellere ayır ve 2D diziye dönüştür
pixels = image.reshape(-1, 3).astype(np.float32)

# Mean Shift algoritmasını uygula
bandwidth = 30  # Bandwidth değeri
centroids = mean_shift(pixels, bandwidth)

# Her pikseli en yakın centroid ile eşleştir
distances = np.sqrt(((pixels - centroids[:, np.newaxis]) ** 2).sum(axis=2))
labels = np.argmin(distances, axis=0)

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
