import numpy as np
import cv2
import descriptors as desc
import warnings

# Remove warning from code, sqrt warning is expected on code
warnings.filterwarnings("ignore", category=RuntimeWarning)

DS =[
        ['beach_1.jpg', []]
]

# Create DS structure for names and regions features
# [Filename, Regions Features]
DS_AUX =[
        ['beach_1.jpg', []],
        ['beach_2.jpg', []],
        ['beach_3.jpg', []],
        ['beach_4.jpg', []],
        ['beach_5.jpg', []],
        ['boat_1.jpg', []],
        ['boat_2.jpg', []],
        ['boat_3.jpg', []],
        ['boat_4.jpg', []],
        ['boat_5.jpg', []],
        ['cherry_1.jpg', []],
        ['cherry_2.jpg', []],
        ['cherry_3.jpg', []],
        ['cherry_4.jpg', []],
        ['cherry_5.jpg', []],
        ['crater_1.jpg', []],
        ['crater_2.jpg', []],
        ['crater_3.jpg', []],
        ['crater_4.jpg', []],
        ['crater_5.jpg', []],
        ['pond_1.jpg', []],
        ['pond_2.jpg', []],
        ['pond_3.jpg', []],
        ['pond_4.jpg', []],
        ['pond_5.jpg', []],
        ['stHelens_1.jpg', []],
        ['stHelens_2.jpg', []],
        ['stHelens_3.jpg', []],
        ['stHelens_4.jpg', []],
        ['stHelens_5.jpg', []],
        ['sunset1_1.jpg', []],
        ['sunset1_2.jpg', []],
        ['sunset1_3.jpg', []],
        ['sunset1_4.jpg', []],
        ['sunset1_5.jpg', []],
        ['sunset2_1.jpg', []],
        ['sunset2_2.jpg', []],
        ['sunset2_3.jpg', []],
        ['sunset2_4.jpg', []],
        ['sunset2_5.jpg', []]
]

def load_features_DS():
    # DS composed by [filename, descriptors]
    for image in DS:
        img = cv2.imread('../input/' + image[0])
        image[1] = desc.get(img)
        print(image[0] + ' Features obtained')

    print('Dataset Loaded and Regions Descriptors Adquired')

def features_distance(feat1, feat2, w):
    feat_diff = abs(np.array(feat1) - np.array(feat2))

    return np.average(feat_diff, weights=w)

# Regions [size, mean_color, texture, centroid, bounding_box]
def distance(reg1, reg2, w, feat_w):

    size_diff = abs(reg1[0] - reg2[0])
    mean_color_diff = np.mean(abs(reg1[1] - reg2[1]))
    centroid_dist = np.sqrt(np.sum((reg1[3] - reg2[3])**2))
    features_diff = features_distance(reg1[2], reg2[2], feat_w)

    return np.average([size_diff, mean_color_diff, centroid_dist, features_diff], weights=w)

def compare_regions(regions_q, regions_s):
    total = 0
    for reg_q in regions_q:
        # Set high dissimilarity
        minimun = 9999999

        # Compare all the regions
        for reg_s in regions_s:
            dist = distance(reg_q, reg_s, [1, 1, 1, 0], [1, 1, 1, 1, 1])

            if dist < minimun:
                minimun = dist

        total += minimun

    # Returning mean value
    return total/len(regions_q)

def top(img, number):
    # Get regions of image quered
    regions = desc.get(img)

    # Create a rank of dissimilarity with DS
    rank = []
    for image in DS:
        rank.append([image[0], 99999999])

    # Comparing image query with datasets
    for i in range(len(DS)):
        rank[i][1] = compare_regions(regions, DS[i][1])

    sorted(rank, key=lambda x: x[1])

    return rank[:number+1]

load_features_DS()

img = cv2.imread('../input/beach_2.jpg')
top3 = top(img, 3)
print(top3)