import utils
import color

import numpy as np
import cv2

DEBUG = True
ballTrace = [(0, 0)]

def main():

    video = cv2.VideoCapture('../input/input.mp4')


    fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    length = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    width  = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = video.get(cv2.CAP_PROP_FPS)

    output = cv2.VideoWriter('../output/output.mp4', fourcc, fps, (width, height))
    for i in range(length):
        print("I: ", i, "/", length)

        ret, frame = video.read()

        height, width = frame.shape[:2]

        color.detectByColor(frame, 60, ballTrace)

        output.write(frame)

    video.release()
    cv2.destroyAllWindows()

main()
