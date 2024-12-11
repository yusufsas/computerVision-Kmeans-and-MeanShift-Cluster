import cv2
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import os
import time

start = time.time()
image_path = '2283107.jpeg'
# Görüntüyü yükle
image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
print("kmeans clustering")
# Görüntüyü piksellere ayır ve 2D diziye dönüştür
pixels = image.reshape(-1, 3)

# K-Means algoritmasını uygula
k = 2  # Segment sayısı
kmeans = KMeans(n_clusters=k)
kmeans.fit(pixels)
labels = kmeans.predict(pixels)

# Segmentleri yeniden renklendir
segmented_img = kmeans.cluster_centers_[labels]
segmented_img = segmented_img.reshape(image.shape).astype(np.uint8)
cv2.imwrite('segmented.jpg', segmented_img)

end = time.time()
total_time = end - start

# Görüntüyü göster
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title(f"Orijinal Görüntü\nDosya Boyutu: {os.path.getsize(image_path) / 1024:.2f} KB")
plt.imshow(image)
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title(f"K-Means Sonucu\nDosya Boyutu: {os.path.getsize('segmented.jpg') / 1024:.2f} KB\nİşlem Süresi: {total_time:.2f} saniye")
plt.imshow(segmented_img)
plt.axis('off')

plt.show()
