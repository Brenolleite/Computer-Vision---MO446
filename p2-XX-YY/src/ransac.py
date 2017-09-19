import numpy as np
import random

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
        # create matrix A 1x6
        A = np.dot(p1, p2)

        #                     [a d]
        # create matrix A 3x2 [b e]
        #                     [c f]
        A = np.swapaxes(A.reshape(2,3), 0, 1)

        return A
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
        # If not enough matches
        if len(points) <= n_data:
            continue

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

    return A_final