import numpy as np
import cv2
import copy as cp
import scipy.spatial

import sift as SIFT

lowe_ratio = 0.75

def transform_index(indexes, kp1, kp2):


# Returns a list with the matches between descriptor 1 and descriptor 2
def match(desc1, desc2):

    # Array of best matches
    good = []

    # Run over all the descriptors of image 1 matching them with the ones in
    # image 2
    for kptIdx1 in range(desc1.shape[0]):
        # Save the first best match and the second one
        fst = 99999
        snd = 0

        for kptIdx2 in range(desc2.shape[0]):
            # Euclidian distance
            aux = np.linalg.norm(desc1[kptIdx1] - desc2[kptIdx2])

            # Saving the fisrt and second best match and their index
            if aux < fst:
                snd = fst
                fst = aux
                save_index = kptIdx2

        # Checking for the ratio between the first and second match, using the
        # ratio Lowe suggested in his paper
        if fst < lowe_ratio * snd:
            x = []
            x.append(cv2.DMatch(kptIdx1, save_index, fst))
            good.append(x)

    return good

def match_tree(desc1, kp1, desc2, kp2, treshold):
    # Choose better image to be on search tree
    if(len(desc1) > len(desc2)):
        desc_l = desc1
        kp_l = kp1
        desc_s = desc2
        kpt_s = kp2
    else:
        desc_l = desc2
        kp_l = kp2
        desc_s = desc1
        kp_s = kp1

    # Create search tree
    kdtree = scipy.spatial.KDTree(desc_l)

    # Search on tree using euclidian distance
    d, i = kdtree.query(desc_s, 1, distance_upper_bound=treshold)

    # Create tuples with values
    array = np.array((np.arange(len(d)), i, d)).T

    # Clear all the matches over the treshold
    array = array[array[:,2] < treshold]

    transform_index()

    return i


# Debug
img1 = cv2.imread('input/img1.png')
img2 = cv2.imread('input/img2.png')
img3 = np.zeros(img2.shape)

kp1, desc1 = SIFT.sift(cp.copy(img1))
kp2, desc2 = SIFT.sift(cp.copy(img2))

good = match(desc1, desc2)

img3 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, good, img3, flags=2)
cv2.imwrite('debug/matches.png', img3)