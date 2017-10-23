import cv2
import numpy as np

def conn_comp(img, connectivity):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    nlabels, labels, stats, centroids = cv2.connectedComponentsWithStats(img, connectivity, cv2.CV_32S)

    return labels, centroids
