import logging
import os

import numpy as np

logger = logging.getLogger(__name__)

FLOAT_NUMPY = np.float32

try:
    os.mkdir("dataset")
except:
    pass

DATASET_PATH = os.getcwd() + "/dataset"
