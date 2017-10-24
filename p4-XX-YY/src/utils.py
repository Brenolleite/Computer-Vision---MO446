import time as t
import math
import numpy as np
import copy as cp
import cv2
from random import randint

class Time:
    # Measuring time
    def __init__(self):
        self.start_time = t.time()

    def elapsed(self):
        return t.time() - self.start_time


# Save kmeans colored image
def k_image(img, center, label):
    center = np.uint8(center)
    aux = center[label.flatten()]
    kimg = aux.reshape((img.shape))

    return kimg

# Create image from components
def components_image(labels):
    # Create serial components for coloring
    count = 1
    for i in np.unique(labels):
        if i != -1 and i != 0:
            labels[labels == i] = count
            count += 1

    colors = []

    # Define some colors manually
    colors.append((255,255,255))
    colors.append((0,0,0))
    colors.append((0,255,0))
    colors.append((0,0,255))
    colors.append((255,0,0))
    colors.append((255,0,255))

    # Create random colors
    for i in np.unique(labels) - len(colors):
        colors.append(((randint(0, 255)), randint(0, 255), randint(0, 255)))

    # Create image
    height, width = labels.shape
    img = np.zeros((height, width, 3))

    for y in range(height):
        for x in range(width):
            img[y,x] = colors[labels[y,x] + 1]

    return img.astype(np.uint8)