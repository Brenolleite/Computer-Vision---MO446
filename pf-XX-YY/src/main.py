""" Main file """
import utils
import color
import flow

import cv2
import numpy as np


def start():
    """ Starting function """

    # ----- Flags -----
    webcam = False
    diff_color_flag = False
    hough_flag = True
    motion_flag = True
    kalman_flag = False

    # ------------ Params --------------------
    resize = 0.3
    motion_freq = 5
    input_file = '../input/mixed_shape.mp4'
    output_file = '../output/output.mp4'
    remove_frames = 5
    # ------------ Params --------------------

    # Setup  video file or webcam stream
    if webcam:
        video = cv2.VideoCapture(0)
        length = 0
    else:
        video = cv2.VideoCapture(input_file)

        fourcc = cv2.VideoWriter_fourcc(*'MPEG')
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

    i = 0
    prev_frame = frame
    balls_centroid = np.float32([])
    raw_centroids = []
    balls_trace = []

    # Counter to motion
    started = False

    # Loop that either runs over every video frame or keeps getting the webcam
    # stream
    while i < length or webcam:
        print("\nProgress ", i, "|", length - 1)

        # Informations from each frame
        _, frame = video.read()
        width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Resize frame for better performace
        frame = cv2.resize(frame, (int(width * resize), int(height * resize)))

        # Detect balls using color and measure the time it took to run
        t = utils.Time()
        balls_info = color.detectByColor(frame, hough_flag)
        print("Time to Detect Balls: ", t.elapsed())

        # Saves and draws the "groundthruth of the centroids
        centroids = utils.parseCentroidInfo(balls_info)
        if len(centroids) > 0:
            raw_centroids.append(utils.parseCentroidInfo(balls_info))

        # Removes points each 5 frames
        if i % remove_frames == 0:
            raw_centroids = utils.maintain_size(raw_centroids,
                                                len(raw_centroids) - 1)

        # Keep size and draw motion flow
        raw_centroids = utils.maintain_size(raw_centroids, 100)
        frame = utils.drawMotionFlow(frame, raw_centroids, (0, 0, 255))

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
                balls_centroid, st, err = flow.motion_flow(prev_frame,
                                                           frame,
                                                           balls_centroid)

                # Save the 100s last points and draws them
                balls_trace.append(balls_centroid)

                # Removes points each 5 frames
                if i % remove_frames == 0:
                    balls_trace = utils.maintain_size(balls_trace,
                                                      len(balls_trace) - 1)

                # Keep size and draw motion flow
                balls_trace = utils.maintain_size(balls_trace, 100)
                frame = utils.drawMotionFlow(frame, balls_trace)

        # Draw all the bounding boxes in the frame
        if len(balls_info) > 0:
            frame = utils.drawBallBox(frame, balls_info, diff_color_flag)

        # Output setup
        if webcam:
            # Show the frame in a window
            cv2.imshow('Frame', frame)

            # Await for key input to stop execution
            if cv2.waitKey(1) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                break

        else:
            # Save frames into file
            output.write(frame)

        # Updates previous frame
        prev_frame = frame
        i += 1

    video.release()

start()
