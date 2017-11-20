import cv2
import numpy as np

class Kalman:
    def __init__(self):
        # Init kalman filter
        self.kalman = cv2.KalmanFilter(2, 2)

        # Kalman filter params
        #kalman.measurementMatrix = np.array([[1,0,0,0],[0,1,0,0]], np.float32)
        #kalman.transitionMatrix = np.array([[1,0,1,0],[0,1,0,1],[0,0,1,0],[0,0,0,1]], np.float32)
        #kalman.processNoiseCov = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]], np.float32) * 0.03
        #kalman.measurementNoiseCov = np.array([[1,0],[0,1]], np.float32) * 0.00003

    def predict(self, measurement):
        # Kalman correction
        self.kalman.correct(measurement)

        # Kalman prediction
        pred = self.kalman.predict()

        return pred