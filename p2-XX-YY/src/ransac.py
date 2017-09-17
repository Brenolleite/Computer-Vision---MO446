import numpy as np
import random
import sift as s
import cv2

def create_matrixes(x_data, y_data, sampled):
    A = np.array([])
    row = 0
    for i in sampled:
        x, y  = x_data[i].pt
        A[row] = [x, y, 1, 0, 0, 0]
        row += 1
        A[row] = [0, 0, 0, x, y, 1]
        row += 1

    return A

def fit_model(X, Y):
    xt = np.traspose(X)
    p1 = np.linalg.inv(np.dot(X, xt))
    p2 = np.dot(xt, Y)

    return np.dot(p1, p2)

def model_error(X, Y):
    return ""

def ransac(x_data, y_data, n_data ,n_iterations, treshold, ratio):
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
input = cv2.imread('../input/img2.png')
input2 = cv2.imread('../input/img11.png')

desc = s.sift(input)
desc2 = s.sift(input2)

print(ransac(desc, desc2, 3, 1, 1, 1))