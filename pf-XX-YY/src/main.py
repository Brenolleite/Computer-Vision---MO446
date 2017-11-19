import utils
import color

import numpy as np
import cv2

# ------------ Params --------------------
WEBCAM      = False
RESIZE      = 0.3
input_file  = '../input/same_color.mp4'
output_file = '../output/output.mp4'
# ------------ Params --------------------

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

while (i < length or WEBCAM):
    print("Progress ", i, "|", length - 1)

    _, frame = video.read()
    width   = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height  = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

    frame = cv2.resize(frame, (int(width * RESIZE), int(height * RESIZE)))

    # Detect balls using color and no hough filter
    ballsInfo = color.detectByColor(frame)

    if len(ballsInfo) > 0:
        frame = utils.drawBallBox(frame, ballsInfo)

        # Update traceBalls and draw lines
        #  traceBalls = utils.drawBallTrace(traceBalls, ballsInfo)

    if WEBCAM:
        cv2.imshow('Frame', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break

    else:
        output.write(frame)

    i += 1

video.release()
