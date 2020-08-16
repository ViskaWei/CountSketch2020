import logging
import numpy as np
import numbers
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter

import collections
import matplotlib.pyplot as plt

class Mat():
    def __init__(vec):
        self.vec=vec
    
   