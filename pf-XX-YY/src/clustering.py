import numpy as np
import cv2

def kmeans(img, K):
    # Image prepeparation
    img = img.reshape((-1, 3))
    img = np.float32(img)

    # Define criteria, number of clusters (K) and apply kmeans
    criteria = (cv2.TERM_CRITERIA_MAX_ITER, 4, 1.0)

    # Using kmeans on colors intensity
    aux, label, center = cv2.kmeans(img, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    return center, label