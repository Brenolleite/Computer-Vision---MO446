import numpy as np
import cv2
import utils

def KLT(video_path):
    # Get video length
    video = cv2.VideoCapture(video_path)
    length = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

    # params for ShiTomasi corner detection
    feature_params = dict(maxCorners = 100,
                          qualityLevel = 0.3,
                          minDistance = 7,
                          blockSize = 7 )

    # Parameters for lucas kanade optical flow
    lk_params = dict(winSize  = (15,15),
                     maxLevel = 2,
                     criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    # Get frames
    ret, frame1 = video.read()

    # Transform frame to grayscale
    frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)

    # Get keypoints
    kps = []
    kp = cv2.goodFeaturesToTrack(frame1, mask = None, **feature_params)

    # Remove single dimention to add on table
    kps.append(kp.squeeze())

    for i in range(0, length - 1):
        # Get frames
        ret, frame2 = video.read()

        # Transform frame to grayscale
        frame2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

        # Calculate optical flow
        kp, st, err = cv2.calcOpticalFlowPyrLK(frame1, frame2, kps[i-1], None, **lk_params)

        # Add new points to matrix of frames
        kps.append(kp.squeeze())

        # Search index of bad kp
        st = np.array(st)
        index = np.where(st==0)

        # Delete bad keypoints
        np.delete(kps, index, axis=1)

        # update frame
        frame1 = frame2

        i += 1

    video.release()

    return kps

video_path = '../input/input2.mp4'
kps = KLT(video_path)
utils.videoFlow(kps, video_path, '../output/opencv_flow.avi', (13, 94, 1))
