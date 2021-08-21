import logging


class BasicLogger(logging.Logger):
    def __init__(self):
        super(BasicLogger, self).__init__()
        ch = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(name)s: %(message)s')
        ch.setFormatter(formatter)


def getLogger(name):
    if name:
        return BasicLogger.manager.getLogger(name)
