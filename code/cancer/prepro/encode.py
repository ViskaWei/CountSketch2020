# -*- coding: utf-8 -*-
import sys
import os
import copy
import pickle
import numpy as np
import pandas as pd
from collections import Counter
from timeit import default_timer as timer
from concurrent.futures import ThreadPoolExecutor, as_completed
from sklearn import preprocessing
import warnings
warnings.simplefilter("ignore")
from cancer.prepro.dataset import Dataset
from util.prepro import get_col_norm_pd,get_minmax_pd,get_encode_stream

############## Process Data #################
def process_dataset_pc(dataDir, num, pca_comp,ISSMTH, SMTH, TEST):   
    ds = Dataset(ISSMTH, SMTH)
    ds.load(dataDir, num=1, ini = 0)
    num  = images.N_img  #in case: num > N_images
#     pca_combined = np.zeros([images.layer, images.layer])
    with ThreadPoolExecutor() as executor: 
        futures = []
        for idx in range(num):
            futures.append(executor.submit(lambda x: run_step_multiple(x, images, ISSMTH, SMTH, TEST), idx))
            # print(f" No.{idx} image is loaded")
        mul_comb = np.zeros([images.layer,images.layer])
        for future in as_completed(futures):
            mul = future.result()
            mul_comb += mul
        pc = run_step_pc(mul_comb, pca_comp)
    return images.data1D, pc


def run_step_pc(mul_comb, pca_comp):
    # use svd since its commputational faster
    print(f"=============== PCA N_component: {pca_comp} ===============")
    u,s,v = np.linalg.svd(mul_comb)
    assert np.allclose(u, v.T)
    print('Explained Variance Ratio', np.round(s/sum(s),3))
    pc = u[:,:pca_comp]
    # np.savetxt('eval.txt',
    return pc

def process_pca(data1Ds, pc, num):
    pca_results = run_step_pc_transform(0, data1Ds, pc)
    for i in range(1,num):
        pca_result = run_step_pc_transform(i, data1Ds, pc)
        pca_results = np.vstack((pca_results,pca_result))
    print('========= Intensity ==============')
    intensity = (np.sum(pca_results**2, axis = 1))**0.5
    return intensity, pca_results

def run_step_pc_transform(x, data1Ds, pc):
    return data1Ds[x].dot(pc)

#===============================intensity=====================================
def process_intensity(pca_result, intensity, pca_comp, PREPRO_CUTOFF,ONPCA,ONINT,r=0.01,wdir='.'):
    if PREPRO_CUTOFF:
        # cut=np.mean(intensity)
        cut = run_step_cutoff(intensity) 
        pickle.dump(cut,open(f'{wdir}/cutoff.txt','wb'))
    else: 
        cut=pickle.load(open(f'{wdir}/cutoff.txt','rb')) 
    print('cutoff', cut)
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
        

def run_step_cutoff(intensity, N_bins = 100, N_sigma = 3):
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

def process_rebin(df_norm, base, dtype):
    stream1D=get_encode_stream(df_norm, base,dtype)
    return stream1D



