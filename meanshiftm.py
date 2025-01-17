import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import time
import cv2
from sklearn.cluster import MeanShift, estimate_bandwidth
from joblib import Parallel, delayed

def gaussian_kernel(distance, bandwidth):
    return np.exp(-0.5 * (distance / bandwidth) ** 2)

# OpenCV Mean Shift ile görüntü segmentasyonu
def mean_shift_opencv(path, quantile=0.2, n_samples=100):
    start = time.time()
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    flat_image = img.reshape((-1, 3))
    flat_image = np.float32(flat_image)
    
    # Bandwidth tahmini
    bandwidth = estimate_bandwidth(flat_image, quantile=quantile, n_samples=n_samples)
    
    ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
    ms.fit(flat_image)
    labels = ms.labels_
    cluster_centers = ms.cluster_centers_
    
    segmented_image = labels.reshape(img.shape[:2])
    result_image = np.zeros_like(img)
    for i in range(len(cluster_centers)):
        result_image[segmented_image == i] = cluster_centers[i]
    
    end = time.time()
    total_time = end - start
    
    opencv_img_path = 'opencv_meanshift_img.jpeg'
    cv2.imwrite(opencv_img_path, cv2.cvtColor(result_image, cv2.COLOR_RGB2BGR))
    size = os.path.getsize(opencv_img_path) / 1024
    
    return result_image, total_time, size

# Manuel Mean Shift ile görüntü segmentasyonu
def update_pixel(i, flat_image, bandwidth):
    distances = np.linalg.norm(flat_image - flat_image[i], axis=1)
    weights = gaussian_kernel(distances, bandwidth)
    return np.sum(weights[:, np.newaxis] * flat_image, axis=0) / np.sum(weights)

def mean_shift_manual(path, bandwidth=0.0009, max_iter=1):
    start = time.time()
    img = cv2.imread(path)
    h, w, c = img.shape
    img = cv2.resize(img, (w//5, h//5))  # Görsel boyutunu küçültme
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    flat_image = img.reshape((-1, 3))
    flat_image = np.float32(flat_image)
    
    for i in range(max_iter):
        old_image = flat_image.copy()
        
        # Paralel işlemle pikselleri güncelleme
        flat_image = np.array(Parallel(n_jobs=-1)(
            delayed(update_pixel)(i, flat_image, bandwidth) for i in range(len(flat_image))
        ))

        if np.linalg.norm(old_image - flat_image) < 1e-3:
            break
    
    segmented_image = flat_image.reshape(img.shape)
    end = time.time()
    total_time = end - start
    
    # Sonuçları kaydetme
    manual_img_path = 'manual_meanshift_img.jpeg'
    Image.fromarray(segmented_image.astype(np.uint8)).save(manual_img_path)
    size = os.path.getsize(manual_img_path) / 1024
    
    return segmented_image, total_time, size




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

def kmeans_opencv(path,k,max_iter,epsilon):
    start = time.time()
    img = cv2.imread(path)
    #img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    pikseller = img.reshape((-1,3))
    pikseller = np.float32(pikseller)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,max_iter,epsilon)
    iter, labels, centers = cv2.kmeans(pikseller,k,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)
    ilkmerkezler = centers.copy()
    print(ilkmerkezler)
    centers = np.uint8(centers)
    yeni_pikseller = centers[labels.flatten()]
    yeni_img = yeni_pikseller.reshape(img.shape)
    opencv_img_path = 'opencv_img.jpeg'
    cv2.imwrite(opencv_img_path,yeni_img)
    size = os.path.getsize(opencv_img_path)/1024
    end = time.time()
    total_time = end - start
    return yeni_img,total_time,ilkmerkezler,size
def kmeans_ozel(path,k,max_iter,epsilon,merkez):
    start = time.time()
    image_path = path
    image = Image.open(image_path)
    image = image.convert('RGB')
    image_data = np.array(image)
    h,w,c = image_data.shape
    pikseller = image_data.reshape(-1,3)
    if merkez is None:
        merkezler =  pikseller[:k]
    else:
        merkezler= merkez #pikseller[:k]
    iteration_merkezleri = []
    iterasyon_sayisi = 0
    epsilon = epsilon
    for iteration in range(max_iter):
        iterasyon_sayisi += 1
        distances = np.linalg.norm(pikseller[:, np.newaxis] - merkezler, axis=2)
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
        yeni_merkezler = np.zeros_like(merkezler)
        for i in range(k):
            kume_merkezleri = pikseller[labels == i]
            if len(kume_merkezleri) > 0:
                yeni_merkezler[i]=kume_merkezleri.mean(axis=0)
            else:
                yeni_merkezler[i]=merkezler[i]
        iteration_merkezleri.append(merkezler.copy())
        merkez_mesafesi = np.linalg.norm(merkezler - yeni_merkezler, axis=1).max()
        if merkez_mesafesi < epsilon:
            break
        merkezler = yeni_merkezler
    kmean_pikseller = np.array([merkezler[label] for label in labels],dtype=np.uint8)
    kmean_imaj = kmean_pikseller.reshape(h,w,c)
    kmean_image_path = 'kmean_image.jpeg'
    yeni_pil_image = Image.fromarray(kmean_imaj)
    yeni_pil_image.save(kmean_image_path, 'JPEG')
    yeni_size = os.path.getsize(kmean_image_path) / 1024
    end = time.time()
    total_time = end - start
    return image,kmean_imaj,total_time,yeni_size, iterasyon_sayisi







# Örnek kullanım
image_path = '2283107.jpeg'
opencv_result, opencv_time, opencv_size = mean_shift_opencv(image_path)
manual_result, manual_time, manual_size = mean_shift_manual(image_path)




cv_image,cv_time,cv_merkezler,cv_size = kmeans_opencv(path,k,iterasyon,epsilon)
org_image, kmean_ozel_imaj,ozel_total_time,ozel_yeni_size,ozel_iterasyon =kmeans_ozel(path,k,iterasyon,epsilon,None)
org_image2, kmean_ozel_imaj2,ozel_total_time2,ozel_yeni_size2,ozel_iterasyon2 =kmeans_ozel(path,k,iterasyon,epsilon,cv_merkezler)



# Orijinal görüntü
original_image = cv2.imread(image_path)
original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

# Fark hesaplaması
opencv_diff = np.sum(np.abs(original_image - opencv_result))


manual_result_resized = cv2.resize(manual_result, (original_image.shape[1], original_image.shape[0]))

# Şimdi karşılaştırma yapabilirsiniz
manual_diff = np.sum(np.abs(manual_result_resized))






cust_diff = (np.sum(org_image-kmean_ozel_imaj))
cust_diff2 = (np.sum(org_image-kmean_ozel_imaj2))
cv_diff = (np.sum(org_image-cv_image))




# manual_diff = np.sum(np.abs(original_image-manual_result))
# Sonuçların görselleştirilmesi
plt.figure(figsize=(24, 7))

plt.subplot(1, 7, 1)
plt.title(f"Orijinal Görüntü\nBoyut: {original_image.shape[1]}x{original_image.shape[0]}")
plt.imshow(original_image)
plt.axis('off')

plt.subplot(1, 7, 2)
plt.title(f"OpenCV Mean Shift\nFark: {opencv_diff}\n Süresi: {opencv_time:.2f} saniye")
plt.imshow(opencv_result)
plt.axis('off')

plt.subplot(1, 7, 3)
plt.title(f"Manuel Mean Shift \nFark: {manual_diff}\n Süresi: {manual_time:.2f} saniye")
plt.imshow(manual_result)
plt.axis('off')

# plt.subplot(1, 7, 4)
# plt.title(f"Orijinal Görüntü\nBoyut: {org_image.size[1]}x{org_image.size[0]}")
# plt.imshow(org_image)
# plt.axis('off')

plt.subplot(1, 7, 5)
plt.title(f"Ozel K-Means \nFark: {cust_diff}\n Süresi: {ozel_total_time:.2f} saniye\n iterasyon:{ozel_iterasyon}")
plt.imshow(kmean_ozel_imaj)
plt.axis('off')

plt.subplot(1, 7, 6)
plt.title(f"Opencv K-Means \nFark: {cv_diff} KB\n Süresi: {cv_time:.2f} saniye\n")
plt.imshow(cv_image)
plt.axis('off')

plt.subplot(1, 7, 7)
plt.title(f"Ozel K-Means -2\nfark: {cust_diff2} KB\n Süresi: {ozel_total_time2:.2f} saniye\n iterasyon:{ozel_iterasyon2}")
plt.imshow(kmean_ozel_imaj2)
plt.axis('off')

plt.show()