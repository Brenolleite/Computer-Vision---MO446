import numpy as np
import random


import sift as s
import match as m
import cv2

def create_matrixes(x_data, y_data, sampled):
    X_matrix = np.array([])
    Y_matrix = np.array([])

    for i in sampled:
        x, y  = x_data[i].pt
        X_matrix.append([x, y, 1, 0, 0, 0], axis=0)
        X_matrix.append([0, 0, 0, x, y, 1], axis=0)

        x, y  = y_data[i].pt
        Y_matrix.append([x], axis=0)
        Y_matrix.append([y], axis=0)

    return X_matrix, Y_matrix

def fit_model(X, Y):
    xt = np.traspose(X)
    p1 = np.linalg.inv(np.dot(X, xt))
    p2 = np.dot(xt, Y)

    return np.dot(p1, p2)

def model_error(X, Y):



    return ""

def ransac(matches, x_data, y_data, n_data ,n_iterations, treshold, ratio):
    for i in range(n_iterations):
        # Sample n_data points
        sampled = random.sample(range(len(x_data[0])), 6)

        # Crate matrixes used for the model
        X, Y = create_matrixes(x_data, y_data, sampled)

        # Fit the model with the matrixes
        A = fit_model(X, Y)

        print(A)

    return ""


# Debug
img2 = cv2.imread('../input/img2.png')
img1 = cv2.imread('../input/img1.png')

kp1, desc1 = s.sift(img1)
kp2, desc2 = s.sift(img2)

#matches = m.match(desc1, desc2)

matches_tree = m.match_tree(desc1, kp1, desc2, kp2, 150)

#print(ransac(matches, desc, desc2, 3, 1, 1, 1))