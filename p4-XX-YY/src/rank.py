import numpy as np
import cv2
import descriptors as desc
import warnings
import os.path

# Remove warning from code, sqrt warning is expected on code
warnings.filterwarnings("ignore", category=RuntimeWarning)

# Create DS structure for names and regions features
# [Filename, Regions Features]
DS = [
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

def load_features_DS(file = False, verbose = False):
    print('Loading dataset and descriptors')
    # If using file
    if file:
        filexists = os.path.exists('src/descriptors.data')

        if filexists:
            # Open file and read descriptors
            f = open('src/descriptors.data', 'r')
            for image in DS:
                # Remove string array() for use eval
                image[1] = eval(f.readline().replace(')', '').replace('array(', ''))

                if verbose:
                    print(image[0], 'Loaded')

            print('Dataset Loaded and Regions Descriptors Adquired by File Data')

            # Close and exit
            f.close()
            return
        else:
            f = open('src/descriptors.data', 'w+')

    # DS composed by [filename, descriptors]
    for image in DS:
        img = cv2.imread('input/' + image[0])
        image[1] = desc.get(img)

        # Write descriptors to file
        if file:
            f.write(','.join(str(x) for x in image[1]))
            f.write('\n')

    print('Dataset Loaded and Regions Descriptors Adquired')

    # Close file
    if file:
        print('Descriptors wrote in ''descriptors.data'' inside src folder')
        f.close()

def features_distance(feat1, feat2, feat_w):
    feat_diff = abs(np.array(feat1) - np.array(feat2))

    return np.average(feat_diff, weights=feat_w)

# Regions [size, mean_color, texture, centroid, bounding_box]
def distance(reg1, reg2, dist_w, feat_w):

    # Calculate all the destances between regions
    size_diff = abs(np.array(reg1[0]) - np.array(reg2[0]))
    mean_color_diff = np.mean(abs(np.array(reg1[1]) - (reg2[1])))
    centroid_dist = np.sqrt(np.sum((np.array(reg1[3]) - np.array(reg2[3]))**2))
    features_diff = features_distance(reg1[2], reg2[2], feat_w)

    return np.average([size_diff, mean_color_diff, centroid_dist, features_diff], weights=dist_w)

def compare_regions(regions_q, regions_s, dist_w = None, feat_w = None):
    # If weights are not passed
    # Use default values
    if dist_w == None:
        dist_w = [1, 1, 1, 1]
    if feat_w == None:
        feat_w = [1, 1, 1, 1, 1]

    dist_w = np.array(dist_w)
    feat_w = np.array(feat_w)

    total = 0
    for reg_q in regions_q:
        # Set high dissimilarity
        minimun = 9999999

        # Compare all the regions
        for reg_s in regions_s:
            dist = distance(reg_q, reg_s, dist_w, feat_w)

            if dist < minimun:
                minimun = dist

        total += minimun

    # Returning mean value
    return total/len(regions_q)

def top(name, img, number):
    # Get regions of image quered
    regions = desc.get(img)

    # Create a rank of dissimilarity with DS
    rank = []
    for image in DS:
        rank.append([image[0], 99999999])

    # Comparing image query with datasets
    for i in range(len(DS)):
        # Remove query from DS results
        if name + '.jpg' != DS[i][0]:
            rank[i][1] = compare_regions(regions, DS[i][1], [1, 2, 2, 0], None)

    # Sort array and transform to ndarray
    rank = np.array(sorted(rank, key=lambda x: x[1]))

    # Return just the labels for the images
    return rank[:number, 0]