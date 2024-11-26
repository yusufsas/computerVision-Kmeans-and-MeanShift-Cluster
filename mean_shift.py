import cv2
import numpy as np
from sklearn.cluster import MeanShift, estimate_bandwidth
import matplotlib.pyplot as plt

# Görüntüyü yükle
image = cv2.imread('image.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Görüntüyü piksellere ayır ve 2D diziye dönüştür
pixels = image.reshape(-1, 3)

# Piksel örneklemesi yap
sample_size = 1000  # Örnekleme boyutu
np.random.seed(0)
sample_indices = np.random.choice(pixels.shape[0], sample_size, replace=False)
sample_pixels = pixels[sample_indices]

# Bandwidth tahmini
bandwidth = estimate_bandwidth(sample_pixels, quantile=0.2, n_samples=500)

# Mean Shift algoritmasını uygula
mean_shift = MeanShift(bandwidth=bandwidth, bin_seeding=True)
mean_shift.fit(sample_pixels)
labels = mean_shift.predict(pixels)
cluster_centers = mean_shift.cluster_centers_

# Segmentleri yeniden renklendir
segmented_img = cluster_centers[labels]
segmented_img = segmented_img.reshape(image.shape).astype(np.uint8)

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
