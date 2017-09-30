import numpy as np
import cv2

# Return a list of tuples [(x, y)] of each keypoint
def kp(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = np.float32(gray)
    dst = cv2.cornerHarris(gray, 2, 3, 0.04)

    kp = []
    threshold = 0.01 * dst.max()
    for i in range(dst.shape[0]):
        for j in range(dst.shape[1]):
            if dst[i][j] > threshold:
                kp.append((i, j))

    return kp

# DEBUG
video = cv2.VideoCapture('../input/input.mp4')
fourcc = cv2.VideoWriter_fourcc(*'DIVX')
ret, frame = video.read()
kp(frame)
