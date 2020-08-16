import os
import numpy as np
import logging
from scipy.ndimage import gaussian_filter
from concurrent.futures import ThreadPoolExecutor, as_completed

#os.environ['DATASET_PATH'] = r'../../../../../../bki/flatw/M21_1/'

class CellDataset():
    def __init__(self, fileDir, nImg, size = [1004,1344,35]):
        self.filePath=None
        self.nImg=nImg
        self.ini=0
        self.size = size
        self.ver, self.hor, self.layer = self.size
        self.data = {}
        self.cov=None
        self.pc=None
        # ===========================  INITIATING  ================================
        self.get_file_path(fileDir)

        # ===========================  FUNCTIONS  ================================

    def get_file_path(self, fileDir):
        filePath = [os.path.join(fileDir, f) for f in os.listdir(fileDir) if f.endswith('.fw')]
        self.filePath = filePath[self.ini:self.ini + self.nImg]
        logging.info("  Loading # {} image(s) ".format(len(self.filePath)))
    
    def load_ith_item(self,idx, isTest=False, isSmooth=True, smooth_sig=None):
        with open(self.filePath[idx],'rb') as f_id:
            img = np.fromfile(f_id, count=np.prod(self.size),dtype = np.uint16)
            img = np.reshape(img, self.size)
 
            if isTest:
                self.ver, self.hor = 400, 300              
                img = img[-self.ver:,-self.hor:,:]
            logging.info("Loaded dataset with shapes: {} {}".format(self.ver,self.hor))

            if isSmooth: 
                img=gaussian_filter(img,sigma=smooth_sig)
                logging.info("  Smoothing with sigma:  {}".format(smooth_sig))
            
            img= np.reshape(img, [self.ver*self.hor, self.layer]).astype('float')
            self.data[idx]=img
        return img.T.dot(img)

    def loader(self, isTest, isSmooth, smooth_sig=None):
        return lambda x: self.load_ith_item(x, isTest, isSmooth, smooth_sig)
     
    def load(self, loader, parallel=True):
        if parallel:
            with ThreadPoolExecutor() as executor: 
                    futures = []
                    for idx in range(self.nImg):
                        futures.append(executor.submit(loader, idx))
                        # print(f" No.{idx} image is loaded")
                    self.cov = np.zeros([self.layer,self.layer])
                    for future in as_completed(futures):
                        mul = future.result()
                        self.cov += mul
        else:
            raise "non parallel"

    def get_pc(self, dim):
        # use svd since its commputational faster
        print(f"=============== PCA: {dim} ===============")
        u,s,v = np.linalg.svd(self.cov)
        assert np.allclose(u, v.T)
        print('Explained Variance Ratio', np.round(s/sum(s),3))
        self.pc = u[:,:dim]

    def get_bulk(self):
        vecs=[self.data[i].dot(self.pc) for i in range(self.nImg)]
        # logging.info(" Bulk Shape:  {}".format(len(vecs)))
        vecs= np.concatenate(vecs)
        logging.info(" Bulk Shape:  {}".format(vecs.shape))
        return vecs 

    # print('========= Intensity ==============')
    # intensity = (np.sum(pca_results**2, axis = 1))**0.5
    # return intensity, pca_results