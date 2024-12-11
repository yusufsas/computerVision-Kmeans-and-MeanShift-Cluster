import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import time

start = time.time()
def distance(p1, p2):
    return np.sqrt(np.sum(np.square(p1 - p2)))

image_path = '2283107.jpeg'
image = Image.open(image_path)
image = image.convert('RGB')
image_data = np.array(image)
h,w,c = image_data.shape
pixels = image_data.reshape(-1,3)

k = 4
max_iterations = 100
centroids= pixels[[3,600,1200,1300]]
iteration_centroids = []


for iteration in range(max_iterations):
    distances = np.linalg.norm(pixels[:, np.newaxis] - centroids, axis=2)
    labels = np.argmin(distances, axis=1)
    # normal dongu
    #labels = []
    #for pixel in pixels:
    #    distances = []
    #    for centroid in centroids:
    #        distances.append(distance(pixel, centroid))
    #    labels.append(np.argmin(distances))
    #labels = np.array(labels)
    # normal dongu bitis

    new_centroids = np.zeros_like(centroids)
    for i in range(k):
        cluster_points = pixels[labels == i]
        if len(cluster_points) > 0:
            new_centroids[i]=cluster_points.mean(axis=0)
        else:
            new_centroids[i]=centroids[i]
    iteration_centroids.append(centroids.copy())

    if np.all(centroids == new_centroids):
        break
    centroids = new_centroids

compressed_pixels = np.array([centroids[label] for label in labels],dtype=np.uint8)
compressed_image = compressed_pixels.reshape(h,w,c)
compressed_image_path = 'compressed_image.jpeg'
compressed_pil_image = Image.fromarray(compressed_image)
compressed_pil_image.save(compressed_image_path, 'JPEG')
compressed_size = os.path.getsize(compressed_image_path) / 1024

end = time.time()
total_time = end - start

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title(f"Orijinal Görüntü\nBoyut: {h}x{w}\nDosya Boyutu: {os.path.getsize(image_path) / 1024:.2f} KB")
plt.imshow(image)
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title(f"K-Means Sonucu\nDosya Boyutu: {compressed_size:.2f} KB\nİşlem Süresi: {total_time:.2f} saniye")
plt.imshow(compressed_image)
plt.axis('off')

plt.show()
