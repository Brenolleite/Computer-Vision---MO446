import time as t
import math
import numpy as np
import copy as cp
import cv2
import random

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
