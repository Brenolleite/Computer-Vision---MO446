import cv2
import rank
import clustering
import connected_comp as cc
import measurements
import utils
import copy as cp

img = cv2.imread('input/cherry_1.jpg')

 # Using kmeans to create clusters
center, labels = clustering.kmeans(img, 5)

# Generate image from clusters
kimg = utils.k_image(img, center, labels)

# Save k-colors image
cv2.imwrite('output/p4-3-0.jpg', kimg)

# Get connected components
components = cc.conn_comp(labels, img)

# Save components image
cv2.imwrite('output/p4-3-1.jpg', utils.components_image(components))

# Get centroids from components
centroids = cc.centroid(components)

# Getting region information
regions = measurements.region_info(img, components, centroids)

# Save bounding boxes
cv2.imwrite('output/p4-3-2.jpg', utils.drawBoundingBox(cp.copy(img), regions))

cv2.imwrite('output/p4-3-3.jpg', utils.drawCentroids(cp.copy(img), regions))

print("Starting macthing process")

# Load all the data from images, and their descriptors
rank.load_features_DS(True)

# Create vector with tests
vec = ['beach_2', 'boat_5', 'cherry_3', 'pond_2', 'stHelens_2', 'sunset1_2', 'sunset2_2']

for image in vec:
    img = cv2.imread('input/' + image + '.jpg')
    print('Query: {0} \t -> Results: {1}'.format(image, rank.top(image, img, 3)))
