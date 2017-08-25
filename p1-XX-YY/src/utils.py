import time as t

class time:
    # Measuring time
    time_started = 0
    start_time = -1

    def __init__(self):
        self.start_time = t.time()
        self.time_started = 1

    def elapsed(self):
        return str(t.time() - self.start_time)
