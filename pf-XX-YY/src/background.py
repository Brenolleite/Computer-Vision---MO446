import cv2

class Background:
    def __init__(self):
        self.fgbg = cv2.bgsegm.createBackgroundSubtractorMOG()

    def subtraction(self, frame):
        return self.fgbg.apply(frame)