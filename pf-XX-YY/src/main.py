import utils
import color

import numpy as np
import cv2

WEBCAM      = True
bBoxArray  = [(0, 0)]
input_file  = '../input/input.mp4'
output_file = '../output/output.mp4'

def main():

    video = None
    length = -1
    i = 0

    if WEBCAM:
        video = cv2.VideoCapture(0)
    else:
        video = cv2.VideoCapture(input_file)

        fourcc  = cv2.VideoWriter_fourcc(*'DIVX')
        length  = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        width   = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        height  = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps     = video.get(cv2.CAP_PROP_FPS)

        output = cv2.VideoWriter(output_file, fourcc, fps, (width, height))

    while (i < length or WEBCAM):
        print("Progress ", i, "|", length)

        _, frame = video.read()

        bBoxArray = color.detectByColor(frame)

        frame = utils.drawBallTrace(frame, bBoxArray)

        if WEBCAM:
            cv2.imshow('Frame', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        else:
            output.write(frame)

        i += 1

    video.release()
    cv2.destroyAllWindows()

main()
