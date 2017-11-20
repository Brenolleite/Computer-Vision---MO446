import utils
import color
import kalman as k
import numpy as np
import cv2

# ------------ Params --------------------
WEBCAM      = False
RESIZE      = 0.3
kalman_flag = True
input_file  = '../input/unique_color.mp4'
output_file = '../output/kalman.mp4'
# ------------ Params --------------------

# Create kalman filters for balls
kalman_filters = {}
colors = color.hsvColor
for c in colors:
    kalman_filters[c] = k.Kalman()
traceKalman = []

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
    ballsInfo = color.detectByColor(frame, True)

    # If using kalman filter
    if kalman_flag:

        # Aplly kalman filter to all balls
        for ball in ballsInfo:
            # Get color and position
            c = ball[0]
            pos = ball[5:7]

            # Corret and predict using kalman
            pred = kalman_filters[c].predict(pos)

            # Checking distance for prediction (threshold)
            d = utils.dist(pred, np.array(pos))
            print(d)
            if d < 30:
                # Creating trace to kalman filter
                traceKalman.append(pred)

                # Drawing points to kalman filter
                utils.drawPoints(frame, traceKalman)

    if len(ballsInfo) > 0:
        # Call the function to draw ball bounding box
        frame = utils.drawBallBox(frame, ballsInfo, True)

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
