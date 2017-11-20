import cv2
import numpy as np

frame = np.zeros((400,400,3), np.uint8)

class Kalman:
    def __init__(self):
        # Init kalman filter
        self.kalman = cv2.KalmanFilter(4, 2)

        # Kalman filter params
        self.kalman.measurementMatrix = np.array([[1,0,0,0],[0,1,0,0]], np.float32)
        self.kalman.transitionMatrix = np.array([[1,0,1,0],[0,1,0,1],[0,0,1,0],[0,0,0,1]], np.float32)
        self.kalman.processNoiseCov = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]], np.float32) * 0.03

    def predict(self, meas):
        # Get x and y from measurement
        x, y = meas

        # Transform measurement to kalman input
        meas = np.array([[np.float32(x)],[np.float32(y)]])

        # Kalman correction
        self.kalman.correct(meas)

        # Kalman prediction
        pred = self.kalman.predict()

        return [int(pred[0]), int(pred[1])]