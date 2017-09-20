import numpy as np
import sift as s
import match as m
import transform as t
import utils as u
import cv2
import copy as cp
import ransac as model

# Creating video
video = cv2.VideoCapture('input/p2-3-4.mp4')
fourcc = cv2.VideoWriter_fourcc(*'DIVX')

# Video params
length = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
width  = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps    = video.get(cv2.CAP_PROP_FPS)
output_video = cv2.VideoWriter('output/p2-3-4.avi',fourcc, fps, (width*2, height))


# create video stabilization
i = 1
ret, frame = video.read()
while(i < length):
    print("Frames {0} - {1}".format(i, i+1))
    ret, frame1 = video.read()
    original = frame1

    print("SIFT")
    kp1, desc1 = s.sift(cp.copy(frame))
    kp2, desc2 = s.sift(cp.copy(frame1))

    print("Matches")
    dmatches_tree = m.match_tree(desc1, desc2, 50)

    print("Ransac")
    tranformation = model.ransac(dmatches_tree, kp1, kp2, 3, 1000, 100, 5)

    if type(tranformation) is np.ndarray:
        print("Transformation")
        frame = t.transform(frame1, tranformation).astype(np.uint8)
    else:
        frame = frame1.astype(np.uint8)

    # Create frame to video
    frame_video = np.concatenate((original, frame), axis=1)

    output_video.write(frame_video)
    i += 1

video.release()
output_video.release()
