import logging

from .logger import setup_logger


class Config:
    BOUNDARY_MARGIN = 0.01
    VISIBILITY_TOLERANCE = 1e-10

    ENABLE_CACHING = True
    MAX_CACHE_SIZE = 1000

    NODE_SIZE = 10
    EDGE_SIZE = 0.5
    PATH_COLOR = "red"
    PATH_WIDTH = 3

    SEA_COLOR = "lightblue"
    LAND_COLOR = "lightgreen"

    def __init__(self, name, verbose=False):
        self.logger = setup_logger(name)
        self.verbose = verbose

    def __setattr__(self, name, value):
        super().__setattr__(name, value)
        if name == "verbose":
            level = logging.INFO if value else logging.WARNING
            self.logger.setLevel(level)
            for handler in self.logger.handlers:
                handler.setLevel(level)
