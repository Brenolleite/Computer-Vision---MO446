import cv2
import utils as ut
import copy as cp

import keypoint as kp
import KLT as klt

#  def test():
#      img = cv2.imread('input/input.png')

#      time = ut.time()
#      kp.sift(img)
#      print("Delta t: ", time.elapsed())

def flowVideo(video, flow, fourcc):

    length = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    width  = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = video.get(cv2.CAP_PROP_FPS)

    flow_video = cv2.VideoWriter('../output/Flow.avi', fourcc, fps, (width, height))
    kp_video = cv2.VideoWriter('../output/Keypoint.avi', fourcc, fps, (width, height))

    for i in range(length):
        ret, frame = video.read()

        kp_img = ut.drawKeypoints(frame, flow[i])
        kp_video.write(kp_img)

video = cv2.VideoCapture('../input/input.mp4')
fourcc = cv2.VideoWriter_fourcc(*'DIVX')

flow = klt.KLT(video, fourcc)

flowVideo(video, flow, fourcc)
