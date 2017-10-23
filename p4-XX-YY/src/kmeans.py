import numpy as np
import cv2
import connected_comp as cc
import utils


def k_means(img):
    Z = img.reshape((-1,3))

    # convert to np.float32
    Z = np.float32(Z)

    # define criteria, number of clusters(K) and apply kmeans()
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    K = 3
    ret, label, center = cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    return center, label


img = cv2.imread('../input/boat_1.jpg')

center, label = k_means(img)

kimg = utils.k_image(img, center, label)

cv2.imwrite('../output/teste.jpg', kimg)

labels, centroids = cc.conn_comp(kimg, 4)
