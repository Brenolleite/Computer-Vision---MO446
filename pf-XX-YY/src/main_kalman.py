import utils
import color
import kalman as k
import numpy as np
import cv2

# ------------ Params --------------------
WEBCAM      = False
RESIZE      = 0.5
kalman_flag = True
input_file  = '../input/occclusion.mp4'
output_file = '../output/kalman.mp4'
# ------------ Params --------------------

# Create kalman filters for balls
kalman_filters = {}
colors = color.hsvColor
for c in colors:
    kalman_filters[c] = (k.Kalman(), [-1, -1])
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
        # Go over all filters
        for c in kalman_filters:
            # Get information for kalman filters
            kalman_filter = kalman_filters[c][0]
            last = kalman_filters[c][1]


            if len(ballsInfo) > 0:
                # Get balls of c color
                kalman_balls = np.array(ballsInfo)[np.where(np.array(ballsInfo)[:,0] == c)[0]][:, 5:7]
            else:
                kalman_balls = []

            if len(kalman_balls) > 0:
                # Get positions to correct in kalman for each ball
                for pos in kalman_balls:
                    # Checking distance for prediction (threshold)
                    d = utils.dist(last, np.array(pos))

                    if d < 30 or last == [-1, -1]:
                        # Corret and predict using kalman
                        kalman_filter.correct(pos)

                # Predict after all corrections
                pred = kalman_filter.predict()
                print("1", pred)
            else:
                # Predict and feed kalman with prediction
                pred = kalman_filter.predict()
                kalman_filter.correct(pred)
                print("2", pred)

            if pred != (0,0):
                # Creating trace to kalman filter
                traceKalman.append(pred)

                # Keep contant size of array
                utils.maintain_size(traceKalman, 50)

                # Drawing points to kalman filter
                utils.drawPoints(frame, traceKalman)


    if len(ballsInfo) > 0:
        # Call the function to draw ball bounding box
        frame = utils.drawBallBox(frame, ballsInfo, False)

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
