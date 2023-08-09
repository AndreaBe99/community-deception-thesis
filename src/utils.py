from enum import Enum
import os

class Utils:
    """Class to store utility functions"""
    def get_device_placement(self):
        """Get device placement, CPU or GPU"""
        return os.getenv("RELNET_DEVICE_PLACEMENT", "CPU")

class FilePaths(Enum):
    """Class to store file paths for data and models"""
    MODELS_DIR = 'models'
    CHKPT_DIR  = 'tmp/rl'

class HyperParams(Enum):
    """
    Hyperparameters for the model.
    """
    WEIGHT = 0.5
