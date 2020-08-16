import os
import numpy as np
import logging
from scipy.ndimage import gaussian_filter
#os.environ['DATASET_PATH'] = r'../../../../../../bki/flatw/M21_1/'

class Dataset():
    def __init__(self, isTest, isSmooth, size = [1004,1344,35]):
        self.size = size
        self.ver, self.hor, self.layer = self.size
        self.data = {}
        self.data1D = {}
        self.test = isTest
        self.isSmooth=isSmooth
        self.smooth_sig = [2.0,2.0,0]

    def load(self, dataDir, num=1, ini = 0):
        data_path = [os.path.join(dataDir, f) for f in os.listdir(dataDir) if f.endswith('.fw')]
        data_path = data_path[ini:ini + num]
        logging.info("  Loading # {} image(s) ".format(len(data_path)))
        with open(data_path[idx],'rb') as f_id:
            data = np.fromfile(f_id, count=np.prod(self.size),dtype = np.uint16)
            self.data[idx] = np.reshape(data, self.size)
 
            if self.test:
                self.ver, self.hor = 400, 300              
                self.data[idx] = self.data[idx][-self.ver:,-self.hor:,:]
            logging.info("Loaded dataset with shapes: {} {}".format(self.ver,self.hor))

            if self.isSmooth: 
                self.data[idx]=gaussian_filter(self.data[idx],sigma=self.smooth_sig)
                logging.info("  Smoothing with sigma:  {}".format(self.smooth_sig))
        
        self.data[idx] = np.reshape(self.data[idx], [self.ver*self.hor, self.layer]).astype('float')
