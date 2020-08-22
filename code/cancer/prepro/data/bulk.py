import logging
import numpy as np
import pandas as pd
import collections

class Bulk():
    def __init__(vecs):
        self.vecs=vecs
        self.intensity=None

        # self.get_intensity()
    
    def get_intensity(self):
        intensity = (np.sum(self.vecs**2, axis = 1))**0.5
        return intensity
    
    def get_cutoff(self, intensity, N_bins = 100, N_sigma = 3):
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

    def get_unit_ball(self, cut):
        mask = intensity > cut
        print('norm length',np.sum(mask))
        mask=mask.astype('bool')
        intencut=intensity[mask]
        df_pca=pd.DataFrame(pca_result[mask],columns=list(range(pca_comp)))
        df_uni= np.divide(df_pca, intencut[:,None])
        df_norm=get_minmax_pd(df_uni,r=r, vmin=None, vmax=None)
        if ONPCA:
            df_p2=get_col_norm_pd(df_pca[[1,2]],r=r,w=False,std=False)
            df_norm=pd.concat([df_p2,df_norm],axis=1)
        if ONINT: 
            intencut=(intencut-np.mean(intencut))/np.std(intencut)
            df_inten=pd.DataFrame(intencut, columns=['int'])
            df_inten=get_col_norm_pd(df_inten,r=r,w=False,std=False)
            df_norm=pd.concat([df_inten,df_norm],axis=1)
        ftr_len=len(df_norm.keys())
        print(df_norm)
        df_norm=pd.DataFrame(df_norm.values, columns=list(range(ftr_len)))
        return df_norm, mask, ftr_len