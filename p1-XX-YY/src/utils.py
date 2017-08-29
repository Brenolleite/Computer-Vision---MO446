import time as t

class time:
    # Measuring time

    def __init__(self):
        self.start_time = t.time()

    def elapsed(self):
        return t.time() - self.start_time
