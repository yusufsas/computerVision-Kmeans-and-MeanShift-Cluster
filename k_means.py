import cv2
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Görüntüyü yükle
image = cv2.imread('image.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Görüntüyü piksellere ayır ve 2D diziye dönüştür
pixels = image.reshape(-1, 3)

# K-Means algoritmasını uygula
k = 4  # Segment sayısı
kmeans = KMeans(n_clusters=k)
kmeans.fit(pixels)
labels = kmeans.predict(pixels)

# Segmentleri yeniden renklendir
segmented_img = kmeans.cluster_centers_[labels]
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
