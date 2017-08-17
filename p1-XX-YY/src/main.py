import cv2
import copy as cp
import numpy as np

input = cv2.imread('./input/input.jpg')

# Gaussian pyramid

def pyrUp(img):
  testImg = cv2.pyrUp(img)

cv2.imwrite('./output/CV2_p1-2-2-0.jpg', img)

#def pyrDown(img):

pyrUP(cp.copy(input))