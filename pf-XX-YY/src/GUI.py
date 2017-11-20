""" GUI creation and manangement """
import cv2

# Color of: DIFF_COLOR | HOUGH | MOTION | TRACE | KALMAN
BGR_COLORS = [(0, 255, 0), (0, 255, 255), (127, 0, 255), (255, 255, 0), (0, 127, 255)]

def drawGUI(frame, diff_color_flag, hough_flag, motion_flag, trace_flag, kalman_flag):
    height, width, _ = frame.shape

    if diff_color_flag:
        frame = cv2.putText(frame, "DiffColor *", (width - 60, 10), cv2.FONT_HERSHEY_TRIPLEX, 0.3, (0, 255, 0))
    if hough_flag:
        frame = cv2.putText(frame, "Hough *", (width - 60, 20), cv2.FONT_HERSHEY_TRIPLEX, 0.3, (0, 255, 255))
    if motion_flag:
        frame = cv2.putText(frame, "Motion *", (width - 60, 30), cv2.FONT_HERSHEY_TRIPLEX, 0.3, (127, 0, 255))
    if trace_flag:
        frame = cv2.putText(frame, "Trace *", (width - 60, 40), cv2.FONT_HERSHEY_TRIPLEX, 0.3, (255, 255, 0))
    if kalman_flag:
        frame = cv2.putText(frame, "Kalman *", (width - 60, 50), cv2.FONT_HERSHEY_TRIPLEX, 0.3, (0, 127, 255))
