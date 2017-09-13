import numpy as np
import cv2

# Return the descriptors found in the image
def sift(img):

    # Convert image to black and white
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Get the keypoints and descriptors
    sift = cv2.xfeatures2d.SIFT_create()
    keyPoints, descriptors = sift.detectAndCompute(gray, None)

    # Debug
    # Draw the keypoints in the original image
    cv2.drawKeypoints(gray, keyPoints, img)
    cv2.imwrite('debug/KeyPoints.png', img)

    # Return the descriptors
    return (keyPoints, descriptors)

# Debug
input = cv2.imread('input/img2.png')

sift(input)
