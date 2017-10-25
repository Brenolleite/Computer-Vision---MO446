import numpy as np
import cv2
import connected_comp as cc
import measurements as mes
import utils

def k_means(img):
    # Image prepeparation
    img = img.reshape((-1, 3))
    img = np.float32(img)

    # Define criteria, number of clusters (K) and apply kmeans
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)

    # Number of Clusters
    K = 2

    # Using kmeans on colors intensity
    aux, label, center = cv2.kmeans(img, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    return center, label

# Debug
img = cv2.imread('../input/boat_1.jpg')

center, label = k_means(img)

kimg = utils.k_image(img, center, label)

cv2.imwrite('../output/teste_kmeans.jpg', kimg)

labels, centroids = cc.conn_comp(kimg, 8)

mes.region_info(img, labels, centroids)
