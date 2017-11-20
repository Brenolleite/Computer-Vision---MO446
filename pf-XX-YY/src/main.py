""" Main file """
import utils
import color
import flow
import kalman as k
import GUI
import cv2
import numpy as np


def start():
    """ Starting function """

    # ----- Flags -----
    webcam          = False
    diff_color_flag = False
    hough_flag      = True
    motion_flag     = True
    trace_flag      = True
    kalman_flag     = True

    # ------------ Params --------------------
    # Video params
    resize = 0.3
    input_file = '../input/collision_diff_color_fail1.mp4'
    output_file = '../output/output.avi'

    # Motion params
    motion_freq = 5

    # Drawing params
    remove_frames = 5
    number_points = 50
    # ------------ Params --------------------

    # Setup  video file or webcam stream
    if webcam:
        video = cv2.VideoCapture(0)
        length = 0
    else:
        video = cv2.VideoCapture(input_file)

        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        length = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = video.get(cv2.CAP_PROP_FPS)

        output = cv2.VideoWriter(output_file,
                                 fourcc,
                                 fps,
                                 (int(width * resize), int(height * resize)))

    # Reads the first frame out of the loop
    _, frame = video.read()

    # Create kalman filters for balls
    kalman_filters = {}
    colors = color.hsvColor
    for c in colors:
        kalman_filters[c] = (k.Kalman(), [-1, -1])
    traceKalman = []

    i = 0
    prev_frame = frame
    balls_centroid = np.float32([])
    raw_centroids = []
    balls_trace = []

    # Counter to motion
    started = False

    # Loop that either runs over every video frame or keeps getting the webcam
    while i < length - 1 or webcam:
        print("Progress ", i + 1, "|", length - 1)

        # Informations from each frame
        _, frame = video.read()
        width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Resize frame for better performace
        frame = cv2.resize(frame, (int(width * resize), int(height * resize)))

        # Detect balls using color and measure the time it took to run
        #t = utils.Time()
        balls_info = color.detectByColor(frame, hough_flag)
        #print("Time to Detect Balls: ", t.elapsed())

        # If using kalman filter
        if kalman_flag:
            # Go over all filters
            for c in kalman_filters:
                # Get information for kalman filters
                kalman_filter = kalman_filters[c][0]
                last = kalman_filters[c][1]


                if len(balls_info) > 0:
                    # Get balls of c color
                    kalman_balls = np.array(balls_info)[np.where(np.array(balls_info)[:,0] == c)[0]][:, 5:7]
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
                else:
                    # Predict and feed kalman with prediction
                    pred = kalman_filter.predict()
                    kalman_filter.correct(pred)

                if pred != (0,0):
                    # Creating trace to kalman filter
                    traceKalman.append(pred)

                    # Keep contant size of array
                    traceKalman = utils.maintain_size(traceKalman, number_points)

                    # Removes points each 5 frames
                    if i % remove_frames == 0:
                        traceKalman = utils.maintain_size(traceKalman,
                                                          len(traceKalman) - 1)


                    # Drawing points to kalman filter
                    utils.drawPoints(frame, traceKalman, GUI.BGR_COLORS[4])

        # Saves and draws the "groundthruth of the centroids
        centroids = utils.parseCentroidInfo(balls_info)
        if len(centroids) > 0:
            raw_centroids.append(utils.parseCentroidInfo(balls_info))

        # Removes points each 5 frames
        if i % remove_frames == 0:
            raw_centroids = utils.maintain_size(raw_centroids,
                                                len(raw_centroids) - 1)

        # Keep size and draw motion flow
        raw_centroids = utils.maintain_size(raw_centroids, number_points)

        if trace_flag:
            frame = utils.drawMotionFlow(frame, raw_centroids, GUI.BGR_COLORS[3])

        # All code related to motion flow
        if motion_flag:
            # Every motion_freq frame resamples the centroids to correct the
            # motion error
            if i % motion_freq == 0 or not started:
                started = False

                # Parse centroid
                balls_centroid = utils.parseCentroidInfo(balls_info)

                # Just start if any ball was found
                if len(balls_centroid) > 0:
                     # Set started to true
                    started = True

            # Calculate the motion flow
            if len(balls_centroid) > 0:

                # Get new centroid using flow
                balls_centroid, _, _ = flow.motion_flow(prev_frame,
                                                        frame,
                                                        balls_centroid)

                # Save the 100s last points and draws them
                balls_trace.append(balls_centroid)

                # Removes points each 5 frames
                if i % remove_frames == 0:
                    balls_trace = utils.maintain_size(balls_trace,
                                                      len(balls_trace) - 1)

                # Keep size and draw motion flow
                balls_trace = utils.maintain_size(balls_trace, number_points)
                frame = utils.drawMotionFlow(frame, balls_trace, GUI.BGR_COLORS[2])

        # Draw all the bounding boxes in the frame
        if len(balls_info) > 0:
            frame = utils.drawBallBox(frame, balls_info, diff_color_flag)

        # Output setup
        if webcam:
            GUI.drawGUI(frame, diff_color_flag, hough_flag, motion_flag, trace_flag, kalman_flag)
            # Show the frame in a window
            cv2.imshow('Frame', frame)

            # Await for key input
            key = cv2.waitKey(1) &0xFF
            if key == ord('q'):
                cv2.destroyAllWindows()
                break
            elif key == ord('h'):
                hough_flag = not hough_flag
            elif key == ord('m'):
                motion_flag = not motion_flag
            elif key == ord('k'):
                kalman_flag = not kalman_flag
            elif key == ord('d'):
                diff_color_flag = not diff_color_flag
            elif key == ord('t'):
                trace_flag = not trace_flag

        else:
            # Save frames into file
            output.write(frame)

        # Updates previous frame
        prev_frame = frame
        i += 1

    video.release()

start()
