import numpy as np
import cv2
from sklearn.cluster import MeanShift, estimate_bandwidth
import matplotlib.pyplot as plt
import os

def mean_shift_segmentation(image_path, quantile=0.2, n_samples=500, bin_seeding=True):
    """
    Mean Shift algoritması ile görüntü segmentasyonu yapar.

    Args:
        image_path (str): Segmentasyonu yapılacak görüntünün yolu.
        quantile (float): Bandwidth tahmini için kullanılan quantile değeri.
        n_samples (int): Bandwidth tahmini için örnek sayısı.
        bin_seeding (bool): Bin seeding kullanılıp kullanılmayacağı.

    Returns:
        segmented_image (ndarray): Segmentasyon sonucu elde edilen görüntü.
        labels (ndarray): Her piksel için etiketler.
    """
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")
    
    # Görüntüyü yükleyin ve RGB formatına dönüştürün
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Failed to load image file: {image_path}")

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Görüntüyü (yükseklik * genişlik, 3) boyutuna getirin
    flat_image = np.reshape(image, [-1, 3])

    # Bandwidth'i tahmin edin
    bandwidth = estimate_bandwidth(flat_image, quantile=quantile, n_samples=n_samples)

    # MeanShift ile segmentasyonu gerçekleştirin
    ms = MeanShift(bandwidth=bandwidth, bin_seeding=bin_seeding)
    ms.fit(flat_image)
    labels = ms.labels_

    # Sonuçları yeniden boyutlandırın
    segmented_image = labels.reshape(image.shape[:2])

    return segmented_image, labels

def plot_segmentation_results(original_image_path, segmented_image, labels):
    """
    Orijinal ve segmentasyon sonuçlarını yan yana görselleştirir.

    Args:
        original_image_path (str): Orijinal görüntünün yolu.
        segmented_image (ndarray): Segmentasyon sonucu elde edilen görüntü.
        labels (ndarray): Her piksel için etiketler.
    """
    original_image = cv2.imread(original_image_path)
    original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

    # Etiketleri renklendir
    height, width = segmented_image.shape
    label_image = np.zeros((height, width, 3), dtype=np.uint8)
    unique_labels = np.unique(labels)
    reshaped_labels = labels.reshape(height, width)
    for label in unique_labels:
        mask = reshaped_labels == label
        label_image[mask] = np.random.randint(0, 255, 3)

    plt.figure(figsize=(15, 5))
    
    # Orijinal görüntüyü göster
    plt.subplot(1, 3, 1)
    plt.title('Original Image')
    plt.imshow(original_image)
    plt.axis('off')

    # Segmentasyon sonucu görüntüyü göster
    plt.subplot(1, 3, 2)
    plt.title('Segmented Image')
    plt.imshow(label_image)
    plt.axis('off')

    # Orijinal görüntü ile segmentasyon sonucu arasındaki farkı hesaplayın ve gösterin
    mean_shift_diff = np.sum(np.abs(np.array(original_image, dtype=np.int16) - np.array(label_image, dtype=np.int16)), axis=2)

    plt.subplot(1, 3, 3)
    plt.title('Difference Image')
    plt.imshow(mean_shift_diff, cmap='gray')
    plt.axis('off')

    plt.show()

# Örnek kullanım
image_path = '2283107.jpeg'
segmented_image, labels = mean_shift_segmentation(image_path)
plot_segmentation_results(image_path, segmented_image, labels)
