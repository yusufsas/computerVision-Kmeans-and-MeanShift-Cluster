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

cv_image,cv_time,cv_merkezler,cv_size = kmeans_opencv(path,k,iterasyon,epsilon)
org_image, kmean_ozel_imaj,ozel_total_time,ozel_yeni_size,ozel_iterasyon =kmeans_ozel(path,k,iterasyon,epsilon,None)
org_image2, kmean_ozel_imaj2,ozel_total_time2,ozel_yeni_size2,ozel_iterasyon2 =kmeans_ozel(path,k,iterasyon,epsilon,cv_merkezler)

cust_diff = (np.sum(org_image-kmean_ozel_imaj))
cust_diff2 = (np.sum(org_image-kmean_ozel_imaj2))
cv_diff = (np.sum(org_image-cv_image))

plt.figure(figsize=(24, 6))
plt.subplot(1, 4, 1)
plt.title(f"Orijinal Görüntü\nBoyut: {org_image.size[1]}x{org_image.size[0]}")
plt.imshow(org_image)
plt.axis('off')

plt.subplot(1, 4, 2)
plt.title(f"Ozel K-Means Sonucu\nFark: {cust_diff}\nİşlem Süresi: {ozel_total_time:.2f} saniye\n iterasyon:{ozel_iterasyon}")
plt.imshow(kmean_ozel_imaj)
plt.axis('off')

plt.subplot(1, 4, 3)
plt.title(f"Opencv K-Means Sonucu\nFark: {cv_diff} KB\nİşlem Süresi: {cv_time:.2f} saniye\n")
plt.imshow(cv_image)
plt.axis('off')

plt.subplot(1, 4, 4)
plt.title(f"Ozel K-Means Sonucu-2\nfark: {cust_diff2} KB\nİşlem Süresi: {ozel_total_time2:.2f} saniye\n iterasyon:{ozel_iterasyon2}")
plt.imshow(kmean_ozel_imaj2)
plt.axis('off')

plt.show()

