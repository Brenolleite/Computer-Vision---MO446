import cv2

import utils as u
import keypoint as kp

def test():
    img = cv2.imread('input/input.png')

    time = u.time()
    kp.sift(img)
    print("Delta t: ", time.elapsed())

test()
