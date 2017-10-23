import numpy as np
import cv2
import connected_comp as cc
import utils


def k_means(img):
    # Img prep
    img = img.reshape((-1, 3))
    img = np.float32(img)

    # define criteria, number of clusters(K) and apply kmeans()
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    
    # Number of Clusters
    c_num = 3

    aux, label, center = cv2.kmeans(img, c_num, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    return center, label

# Debug
img = cv2.imread('../input/boat_1.jpg')

center, label = k_means(img)

kimg = utils.k_image(img, center, label)

cv2.imwrite('../output/teste.jpg', kimg)

labels, centroids = cc.conn_comp(kimg, 4)