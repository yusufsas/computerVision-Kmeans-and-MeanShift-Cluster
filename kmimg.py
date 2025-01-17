import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import time
import cv2

k = 4
iterasyon = 100
epsilon = 0.0001
path = '2283107.jpeg'

def distance(p1, p2):
    return np.sqrt(np.sum(np.square(p1 - p2)))

def kmeans_opencv(path, k, max_iter, epsilon):
    start = time.time()
    img = cv2.imread(path)
    pikseller = img.reshape((-1,3))
    pikseller = np.float32(pikseller)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, max_iter, epsilon)
    iter, labels, centers = cv2.kmeans(pikseller, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    ilkmerkezler = centers.copy()
    centers = np.uint8(centers)
    yeni_pikseller = centers[labels.flatten()]
    yeni_img = yeni_pikseller.reshape(img.shape)
    opencv_img_path = 'opencv_img.jpeg'
    cv2.imwrite(opencv_img_path, yeni_img)
    size = os.path.getsize(opencv_img_path) / 1024
    end = time.time()
    total_time = end - start
    return yeni_img, total_time, ilkmerkezler, size

def kmeans_ozel(path, k, max_iter, epsilon, merkez):
    start = time.time()
    image_path = path
    image = Image.open(image_path)
    image = image.convert('RGB')
    image_data = np.array(image)
    h, w, c = image_data.shape
    pikseller = image_data.reshape(-1, 3)
    if merkez is None:
        merkezler = pikseller[:k]
    else:
        merkezler = merkez
    iteration_merkezleri = []
    iterasyon_sayisi = 0
    for iteration in range(max_iter):
        iterasyon_sayisi += 1
        distances = np.linalg.norm(pikseller[:, np.newaxis] - merkezler, axis=2)
        labels = np.argmin(distances, axis=1)
        yeni_merkezler = np.zeros_like(merkezler)
        for i in range(k):
            kume_merkezleri = pikseller[labels == i]
            if len(kume_merkezleri) > 0:
                yeni_merkezler[i] = kume_merkezleri.mean(axis=0)
            else:
                yeni_merkezler[i] = merkezler[i]
        iteration_merkezleri.append(merkezler.copy())
        merkez_mesafesi = np.linalg.norm(merkezler - yeni_merkezler, axis=1).max()
        if merkez_mesafesi < epsilon:
            break
        merkezler = yeni_merkezler
    kmean_pikseller = np.array([merkezler[label] for label in labels], dtype=np.uint8)
    kmean_imaj = kmean_pikseller.reshape(h, w, c)
    kmean_image_path = 'kmean_image.jpeg'
    yeni_pil_image = Image.fromarray(kmean_imaj)
    yeni_pil_image.save(kmean_image_path, 'JPEG')
    yeni_size = os.path.getsize(kmean_image_path) / 1024
    end = time.time()
    total_time = end - start
    return image, kmean_imaj, total_time, yeni_size, iterasyon_sayisi

def mean_shift_opencv(path):
    start = time.time()
    img = cv2.imread(path)
    mean_shift_img = cv2.pyrMeanShiftFiltering(img, sp=70, sr=70)
    mean_shift_img_path = 'mean_shift_img.jpeg'
    cv2.imwrite(mean_shift_img_path, mean_shift_img)
    size = os.path.getsize(mean_shift_img_path) / 1024
    end = time.time()
    total_time = end - start
    return mean_shift_img, total_time, size

# def mean_shift_manual(image, sp=300, sr=600, max_iter=100, epsilon=0.1):
#     img = np.array(image, dtype=np.float32)
#     h, w, c = img.shape
#     shifted = np.copy(img)

#     for y in range(h):
#         for x in range(w):
#             center = img[y, x]
#             for _ in range(max_iter):
#                 y_min = max(y - sp, 0)
#                 y_max = min(y + sp, h - 1)
#                 x_min = max(x - sp, 0)
#                 x_max = min(x + sp, w - 1)

#                 # Using array slicing for region extraction and vectorized calculation
#                 region = img[y_min:y_max+1, x_min:x_max+1]
#                 diff = region - center
#                 mask = np.linalg.norm(diff, axis=2) < sr

#                 if np.sum(mask) == 0:
#                     break

#                 weighted_region = region[mask]
#                 new_center = np.mean(weighted_region, axis=0)

#                 shift_dist = np.linalg.norm(new_center - center)
#                 center = new_center

#                 if shift_dist < epsilon:
#                     break

#             shifted[y, x] = center

#     return np.uint8(shifted)

cv_image, cv_time, cv_merkezler, cv_size = kmeans_opencv(path, k, iterasyon, epsilon)
org_image, kmean_ozel_imaj, ozel_total_time, ozel_yeni_size, ozel_iterasyon = kmeans_ozel(path, k, iterasyon, epsilon, None)
org_image2, kmean_ozel_imaj2, ozel_total_time2, ozel_yeni_size2, ozel_iterasyon2 = kmeans_ozel(path, k, iterasyon, epsilon, cv_merkezler)
mean_shift_img, mean_shift_time, mean_shift_size = mean_shift_opencv(path)

start = time.time()
# manual_mean_shift_img = mean_shift_manual(org_image)
end = time.time()

manual_mean_shift_time = end - start
# manual_mean_shift_img_pil = Image.fromarray(manual_mean_shift_img)
manual_mean_shift_img_path = 'manual_mean_shift_img.jpeg'
# manual_mean_shift_img_pil.save(manual_mean_shift_img_path)
manual_mean_shift_size = os.path.getsize(manual_mean_shift_img_path) / 1024

cust_diff = np.sum(np.array(org_image, dtype=np.int16) - kmean_ozel_imaj)
cust_diff2 = np.sum(np.array(org_image, dtype=np.int16) - kmean_ozel_imaj2)
cv_diff = np.sum(np.array(org_image, dtype=np.int16) - cv_image)
mean_shift_diff = np.sum(np.array(org_image, dtype=np.int16) - mean_shift_img)
# manual_mean_shift_diff = np.sum(np.array(org_image, dtype=np.int16) - manual_mean_shift_img)

plt.figure(figsize=(30, 6))
plt.subplot(1, 6, 1)
plt.title(f"Orijinal Görüntü\nBoyut: {org_image.size[1]}x{org_image.size[0]}")
plt.imshow(org_image)
plt.axis('off')

plt.subplot(1, 6, 2)
plt.title(f"Ozel K-Means Sonucu\nFark: {cust_diff}\nİşlem Süresi: {ozel_total_time:.2f} saniye\nIterasyon: {ozel_iterasyon}")
plt.imshow(kmean_ozel_imaj)
plt.axis('off')

plt.subplot(1, 6, 3)
plt.title(f"Opencv K-Means Sonucu\nFark: {cv_diff}\nİşlem Süresi: {cv_time:.2f} saniye")
plt.imshow(cv_image)
plt.axis('off')

plt.subplot(1, 6, 4)
plt.title(f"Ozel K-Means Sonucu-2\nFark: {cust_diff2}\nİşlem Süresi: {ozel_total_time2:.2f} saniye\nIterasyon: {ozel_iterasyon2}")
plt.imshow(kmean_ozel_imaj2)
plt.axis('off')

plt.subplot(1, 6, 5)
plt.title(f"Mean Shift Sonucu\nFark: {mean_shift_diff}\nİşlem Süresi: {mean_shift_time:.2f} saniye")
plt.imshow(mean_shift_img)
plt.axis('off')

# plt.subplot(1, 6, 6)
# plt.title(f"Manuel Mean Shift Sonucu\nFark: {manual_mean_shift_diff}\nİşlem Süresi: {manual_mean_shift_time:.2f} saniye\nBoyut: {manual_mean_shift_size:.2f} KB")
# plt.imshow(manual_mean_shift_img)
# plt.axis('off')

plt.show()
