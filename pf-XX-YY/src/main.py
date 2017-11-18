import utils
import color

import numpy as np
import cv2

# ------------ Params --------------------
WEBCAM      = False
input_file  = '../input/random_color.mp4'
output_file = '../output/output.mp4'
# ------------ Params --------------------

if WEBCAM:
    video = cv2.VideoCapture(0)
    length = 0
else:
    video = cv2.VideoCapture(input_file)

    fourcc  = cv2.VideoWriter_fourcc(*'DIVX')
    length  = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    width   = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height  = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps     = video.get(cv2.CAP_PROP_FPS)

    output = cv2.VideoWriter(output_file, fourcc, fps, (width, height))


i = 0
while (i < length or WEBCAM):
    print("Progress ", i, "|", length - 1)

    _, frame = video.read()
    width   = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height  = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if WEBCAM:
        frame = cv2.resize(frame, (int(width * 0.3), int(height * 0.3)))

    traceArray = color.detectByColor(frame)

    frame = utils.drawBallTrace(frame, traceArray)

    if WEBCAM:
        cv2.imshow('Frame', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break

    else:
        output.write(frame)

    i += 1

video.release()

def clearTraceArray(array):
    while len(array) > 100:
        array.pop(0)

    return array

video.release()