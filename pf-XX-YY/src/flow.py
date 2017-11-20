""" Functions related to motion flow calculation/estimation """
import utils
import cv2


def motion_flow(prev_frame, frame, balls_centroid):
    """ Calculates the motion using the previous frame """

    lk_params = dict(winSize=(15, 15),
                     maxLevel=2,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                               10,
                               0.03))

    return cv2.calcOpticalFlowPyrLK(prev_frame,
                                    frame,
                                    balls_centroid,
                                    None,
                                    **lk_params)
