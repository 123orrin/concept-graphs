import time

class Timer(object):
    """Timer clas

    Hint: use with `with Timer(name):`
    """

    def __init__(self, name=None):
        self.name = name

    def __enter__(self):
        self.tstart = time.time()

    def __exit__(self, type, value, traceback):
        if self.name:
            print(
                "[%s]" % self.name,
            )
        print("Elapsed: %0.6fs" % (time.time() - self.tstart))
