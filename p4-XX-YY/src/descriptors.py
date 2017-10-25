import clustering
import utils
import connected_comp as cc
import measurements
import numpy as np

# Get feature vector
def get(img):
    # Using kmeans to create clusters
    center, labels = clustering.kmeans(img, 2)

    # Generate image from clusters
    kimg = utils.k_image(img, center, labels)

    # Get connected components
    components = cc.conn_comp(labels, img)

    # Get centroids from components
    centroids = cc.centroid(components)

    # Getting region information
    regions = measurements.region_info(img, components, centroids)

    return regions