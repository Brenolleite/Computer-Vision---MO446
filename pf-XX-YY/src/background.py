import cv2

class Background:
    def __init__(self):
        self.fgbg = cv2.bgsegm.createBackgroundSubtractorMOG()

    def substraction(self, frame):
        return self.fgbg.apply(frame)
