import time


class TimeMeter:
    """
    Measure running time.
    """

    def __init__(self, frequency=1):
        self.frequency = frequency
        self.t0 = time.time()
        self.t = self.t0

    def update(self):
        t = time.time()
        delta_t = t - self.t
        cumulative_t = t - self.t0
        self.t = t
        return delta_t, cumulative_t

    def display(self, counter, message=None, flush=False):
        t = time.time()
        delta_t = t - self.t
        cumulative_t = t - self.t0
        self.t = t

        if counter % self.frequency == 0:
            msg = "\t\t" if message is None else f"\t\t{message} "
            msg = msg + " " * (45 - len(msg))
            print(
                f"{msg}delta time: {delta_t:.2f} cumulative time: {cumulative_t:.2f}",
                flush=flush,
            )

        return delta_t, cumulative_t
