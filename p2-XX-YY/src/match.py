import numpy as np
import cv2
import copy as cp
import scipy.spatial

import sift as SIFT

lowe_ratio = 0.75

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
            # Create OpenCV Structure DMatch (used to generate images)
            x.append(cv2.DMatch(kptIdx1, save_index, fst))
            good.append(x)

    return good

# Faster implementation of matching process
def match_tree(desc1, desc2, treshold):
    # Create search tree
    kdtree = scipy.spatial.KDTree(desc1)

    # Search on tree using euclidian distance
    d, i = kdtree.query(desc2, 1, distance_upper_bound=treshold)

    # Create tuples with values
    array = np.array((i, np.arange(len(d)), d)).T

    # Clear all the matches over the treshold
    array = array[array[:,2] < treshold]

    # Create OpenCV Structure DMatch (used to generate images)
    matches = []
    for match in array:
        x = []
        x.append(cv2.DMatch(int(match[0]), int(match[1]), match[2]))
        matches.append(x)

    return matches
