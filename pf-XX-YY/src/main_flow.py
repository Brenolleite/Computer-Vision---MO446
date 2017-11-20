import utils
import color
import hough
import utils

import numpy as np
import cv2
import math

# ------------ Params --------------------
WEBCAM      = True
RESIZE      = 0.3
input_file  = '../input/volley-1.mp4'
output_file = '../output/output.mp4'
motionFreq  = 15
# ------------ Params --------------------

def sift(img):
    sift = cv2.xfeatures2d.SIFT_create()
    kst = sift.detect(img, None)

    kp = []
    for i in range(len(kst)):
        x = math.floor(kst[i].pt[0])
        y = math.floor(kst[i].pt[1])
        kp.append((x, y))

    return np.array(kp)

if WEBCAM:
    video = cv2.VideoCapture(0)
    length = 0
else:
    video = cv2.VideoCapture(input_file)

    fourcc  = cv2.VideoWriter_fourcc(*'MPEG')
    length  = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    width   = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height  = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps     = video.get(cv2.CAP_PROP_FPS)

    output = cv2.VideoWriter(output_file, fourcc, fps, (int(width * RESIZE), int(height * RESIZE)))

i = 0
traceBalls = []
# HERE
_, frame = video.read()
prevFrame = frame
motionFrame = frame
prevFrameCircleInfo = []
kp = []
ballsCentroid = np.float32([])
ballsTrace = []

lk_params = dict(winSize = (15, 15),
                 maxLevel = 2,
                 criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

while (i < length or WEBCAM):
    print("\nProgress ", i, "|", length - 1)

    _, frame = video.read()
    width   = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height  = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

    frame = cv2.resize(frame, (int(width * RESIZE), int(height * RESIZE)))
    motionFrame = cv2.resize(motionFrame, (int(width * RESIZE), int(height * RESIZE)))

    t = utils.Time()
    # Detect balls by color using hough transform as filter
    # HERE
    #  ballsInfo, circlesInfo = color.detectByColor(frame, True)

    print("Time to detect balls: {0}".format(t.elapsed()))

    # HERE
     #  ballsInfo = color.detectByColor(frame, True)
    if i % motionFreq == 0:
        ballsInfo = color.detectByColor(frame, True)
        ballsCentroid = utils.parseCentroidInfo(ballsInfo)

    #  ballsInfo = color.detectByColor(frame, True)

    if len(ballsCentroid) > 0:
        ballsCentroid, st, err = cv2.calcOpticalFlowPyrLK(prevFrame, frame, ballsCentroid, None, **lk_params)
    
    ballsTrace.append(ballsCentroid)
    frame = utils.drawMotionFlow(frame, ballsTrace)

    if len(ballsInfo) > 0:
        frame = utils.drawBallBox(frame, ballsInfo)

    if WEBCAM:
        cv2.imshow('Frame', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break

    else:
        output.write(frame)

    # HERE
    prevFrame = frame

    i += 1

video.release()
