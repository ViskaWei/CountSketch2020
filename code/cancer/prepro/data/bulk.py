import logging
import numpy as np
import pandas as pd
import collections

class Bulk():
    def __init__(vecs):
        self.vecs=vecs
        self.cutoff=None
    
    def get_intensity(self):
        intensity = (np.sum(self.vecs**2, axis = 1))**0.5
        return intensity
    
    def get_cutoff(self, N_bins = 100, N_sigma = 3):
        intensity=get_intensity()
        para = np.log(intensity[intensity > 1])
        (x,y) = np.histogram(para, bins = N_bins)
        y = (y[1]-y[0])/2 + y[:-1]
        assert len(x) == len(y)
        x_max =  np.max(x)
        x_half = x_max//2
        mu = y[x == x_max]
        sigma = abs(y[abs(x - x_half).argmin()] -mu)
        cutoff_log = N_sigma* sigma + mu
        cutoff = np.exp(cutoff_log).round()
        return cutoff

    