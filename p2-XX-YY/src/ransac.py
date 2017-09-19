import numpy as np
import random

import sift as s
import match as m
import transform as t
import utils as u
import cv2
import copy as cp

# Transform DMatch to points Fomat: [(x, y), (x', y')]
def transform_dmatch(dmatches, kp1, kp2):
    data = []

    for dmatch in dmatches:
        # Get index of X (original) array
        i1 = dmatch[0].queryIdx
        # Get index of Y (train) array
        i2 = dmatch[0].trainIdx
        # Get positions on keypoints
        data.append([kp2[i2].pt, kp1[i1].pt])

    return np.array(data)

# Create matrixes from the model
def create_matrixes(points, sampled):
    X_matrix = []
    Y_matrix = []
    for i in sampled:
        # x, y
        x, y = points[i][0]
        X_matrix.append([x, y, 1, 0, 0, 0])
        X_matrix.append([0, 0, 0, x, y, 1])

        # x', y'
        x, y = points[i][1]
        Y_matrix.append([x])
        Y_matrix.append([y])

    return np.array(X_matrix), np.array(Y_matrix)

# Execute equation 5 (model)
def fit_model(X, Y):
    try:
        xt = X.transpose()
        p1 = np.dot(xt, X)
        p1 = np.linalg.inv(p1)
        p2 = np.dot(xt, Y)
        return np.dot(p1, p2).reshape(3,2)
    except np.linalg.linalg.LinAlgError as err:
        return None

def transform(A, X):
    X = np.insert(X, 2, 1, axis=1)

    return np.dot(X, A)

# Verify model error
def model_error(X, Y):
    errors = []

    for i in range(len(X)):
        error = (((Y[i][0] - X[i][0]) ** 2) + ((Y[i][1] - X[i][1]) ** 2)) ** 0.5

        errors.append((i, error))

    return np.array(errors), np.mean(np.array(errors)[:,1])

def ransac(dmatches, kp1, kp2, n_data ,n_iterations, treshold, ratio):
    error_min = 100000
    A_final = None

    # Transform OpenCV DMatches into array
    points = transform_dmatch(dmatches, kp1, kp2)

    for i in range(n_iterations):
        # Sample n_data points
        sampled = random.sample(range(len(points)), n_data)
        
        # Crate matrixes used for the model
        X, Y = create_matrixes(points, sampled)

        # Fit the model with the matrixes
        A = fit_model(X, Y)

        # Verify if matrix is inverse
        if type(A) is not np.ndarray:
            continue

        # Get just points not sampled
        non_sampled = np.delete(points, sampled, axis=0)

        # Perform transformation on points not sampled
        predict = transform(A, non_sampled[:, 0])

        # Compute error
        errors, total = model_error(predict, non_sampled[:, 1])

        # Get only erros less than treshold
        inliers = errors[errors[:, 1] < treshold]

        # Verify if model is good enough
        if len(inliers) > ratio:
            # Remove just outliers from data
            indexes = np.concatenate([inliers[:,0].astype(int), sampled])

            # Crate matrixes using all inliers and sampled
            X, Y = create_matrixes(points, indexes)

            # Fit the model with the matrixes (inliers + sampled)
            A = fit_model(X, Y)

            # Verify if matrix is inverse
            if type(A) is not np.ndarray:
                continue

            # Perform transformation on inliers + sampled
            predict = transform(A, points[indexes, 0])

            # Compute error
            errors, total = model_error(predict, points[indexes, 1])

            # Store model
            if error_min > total:
                A_final = A
                error_min = total

    print(A_final)
    print(error_min)
    return A_final


# Debug
video = cv2.VideoCapture('../input/video4.mp4')

i = 1

#while(video.isOpened()):
ret, frame = video.read()
ret, frame1 = video.read()

kp1, desc1 = s.sift(cp.copy(frame))
kp2, desc2 = s.sift(cp.copy(frame1))

dmatches_tree = m.match_tree(desc1, desc2, 50)

img3 = np.zeros(frame1.shape)
img3 = cv2.drawMatchesKnn(frame, kp1, frame1, kp2, dmatches_tree, img3, flags=2)
cv2.imwrite('../debug/matches_VIDEO.png', img3)

tranformation = ransac(dmatches_tree, kp1, kp2, 3, 1, 100, 3)

frame = t.transform(frame1, tranformation)

cv2.imwrite('../output/teste' + str(i) + '.png', frame)
i += 1
