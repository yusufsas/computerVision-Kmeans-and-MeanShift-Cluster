import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
import time
from PIL import Image

# Mean Shift uygulanacak dosya yolu
path = '2283107.jpeg'

# Mean Shift algoritması

def mean_shift_segmentation(path, spatial_radius, color_radius, max_iter):
    start = time.time()

    # Görüntüyü yükle
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Görüntüyü Mean Shift algoritmasına uygun formatta işleme
    shifted = cv2.pyrMeanShiftFiltering(img, sp=spatial_radius, sr=color_radius, maxLevel=max_iter)

    # Sonuç dosyasını kaydet
    mean_shift_img_path = 'mean_shift_img.jpeg'
    cv2.imwrite(mean_shift_img_path, cv2.cvtColor(shifted, cv2.COLOR_RGB2BGR))
    size = os.path.getsize(mean_shift_img_path) / 1024

    end = time.time()
    total_time = end - start

    return img, shifted, total_time, size

# Parametreler
spatial_radius = 30   # Uzamsal (spatial) yarıçap
color_radius = 10     # Renk (color) yarıçap
max_iter = 5          # Maksimum iterasyon seviyesi

# Mean Shift'i uygula
original_img, mean_shift_img, mean_shift_time, mean_shift_size = mean_shift_segmentation(
    path, spatial_radius, color_radius, max_iter
)

# Görselleştirme
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.title(f"Orijinal Görüntü\nBoyut: {original_img.shape[1]}x{original_img.shape[0]}")
plt.imshow(original_img)
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title(f"Mean Shift Sonucu\nBoyut: {mean_shift_size:.2f} KB\nSüre: {mean_shift_time:.2f} saniye")
plt.imshow(mean_shift_img)
plt.axis('off')

plt.show()