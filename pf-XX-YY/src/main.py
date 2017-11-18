import utils
import color

import numpy as np
import cv2
import time

WEBCAM      = False
<<<<<<< HEAD
bBoxArray  = []
=======
>>>>>>> 517b18fe6b9f3b976076e5e506a3f39ffffdc9b8
input_file  = '../input/collision_same_color.mp4'
output_file = '../output/output.mp4'

video = None
length = -1
i = 0

<<<<<<< HEAD
if WEBCAM:
    video = cv2.VideoCapture(0)
else:
    video = cv2.VideoCapture(input_file)
=======
    traceArray = []
    video = None
    length = -1
    i = 0
>>>>>>> 517b18fe6b9f3b976076e5e506a3f39ffffdc9b8

    fourcc  = cv2.VideoWriter_fourcc(*'DIVX')
    length  = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    width   = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height  = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps     = video.get(cv2.CAP_PROP_FPS)

    output = cv2.VideoWriter(output_file, fourcc, fps, (width, height))

while (i < length or WEBCAM):
    print("Progress ", i, "|", length - 1)

    _, frame = video.read()
    width   = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height  = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

<<<<<<< HEAD
    if WEBCAM:
        frame = cv2.resize(frame, (int(width * 0.3), int(height * 0.3)))

    bBoxArray = color.detectByColor(frame)
=======
        traceArray = color.detectByColor(frame)

        frame = utils.drawBallTrace(frame, traceArray)
>>>>>>> 517b18fe6b9f3b976076e5e506a3f39ffffdc9b8

    frame = utils.drawBallTrace(frame, bBoxArray)

<<<<<<< HEAD
    if WEBCAM:
        cv2.imshow('Frame', frame)
=======
            if cv2.waitKey(1) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                break
>>>>>>> 517b18fe6b9f3b976076e5e506a3f39ffffdc9b8

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    else:
        output.write(frame)

<<<<<<< HEAD
    i += 1
=======
    video.release()

def clearTraceArray(array):
    while len(array) > 100:
        array.pop(0)

    return array
>>>>>>> 517b18fe6b9f3b976076e5e506a3f39ffffdc9b8

video.release()
cv2.destroyAllWindows()
