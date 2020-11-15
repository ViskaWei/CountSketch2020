# import getpass
import os
import numpy as np
import pandas as pd
import copy
import umap
from collections import Counter
import getpass
from SciServer import Authentication, CasJobs
from sklearn import preprocessing
import warnings
import time
warnings.simplefilter("ignore")
# model_dict={}
from sdss.prepro.label import get_stellar_pd,get_subclass_pd
from util.prepro import get_col_norm_pd, get_std_pd


def get_prepro_std_pds(df,ftr=None, lbl_str=['class','subclass'],sig=3.0): 
    df = df[(df['class']=='STAR' )|(df['class']=='QSO')]    
    if ftr is None:
        df_data=df.drop(columns=lbl_str)
    else:
        df_data=df[ftr]
    df_lbl=df[lbl_str]
    vmean,vstd=df_data.mean(),df_data.std()
    df_snorm=get_std_pd(df_data, vmean, vstd, sig)
    df_lbl=df_lbl.loc[df_snorm.index]
    assert len(df_lbl)==len(df_snorm)
    df_lbl=get_subclass_pd(df_lbl,LBL_PATH='../data/sdss_stars/label.csv')    
    return df_snorm, vmean,vstd,df_lbl

def prepro_std_specs(SPEC_DATA, ftr=None, sig=3.0, w=True,wpath=None):
    df_full=pd.read_csv(SPEC_DATA)
    df_snorm,vmean,vstd,df_label=get_prepro_std_pds(df_full,ftr=ftr, lbl_str=['class','subclass'],sig=sig)
    print(df_snorm.head(),vmean,vstd)
    print('label',df_label.head() )
    if w:
        print(f'writing to {wpath}')
        np.savetxt(f'{wpath}/vmean.txt',vmean)
        np.savetxt(f'{wpath}/vstd.txt',vstd)
        df_snorm.to_csv(f'{wpath}/spec_norm.csv',index=False)
        df_label.to_csv(f'{wpath}/spec_lbl.csv', index=False)
    return df_snorm,vmean,vstd,df_label

def prepro_std_photos(PHOTO_DATA, vmean, vstd, ftr=None, sig=3.0):
    df_full=pd.read_csv(PHOTO_DATA)
    df_data=df_full if ftr is None else df_full[ftr]
    df_snorm=get_std_pd(df_data, vmean, vstd, sig)
    print(df_snorm.shape , df_full.shape)    
    return df_snorm


def get_prepro_pds(df_full,ftr, r=0.01):
    df_norm,vmin,vrng=get_col_norm_pd(df_full[ftr], r=r,w=True,std=False)
    df_label=get_stellar_pd(df_full[['class','subclass','lu']])    
    return df_norm,vmin,vrng,df_label


def prepro_specs(SPEC_DATA, ftr, r=0.01,w=True,wpath=None):
    dfspec=pd.read_csv(SPEC_DATA)
    dfspec_norm,vmin,vrng,df_label=get_prepro_pds(dfspec,ftr,r=r)
    print(dfspec_norm.head(),vmin,vrng)
    print('label',df_label.head() )
    if w:
        print(f'writing to {wpath}')
        np.savetxt(f'{wpath}/vmin.txt',vmin)
        np.savetxt(f'{wpath}/vrng.txt',vrng)
        dfspec_norm.to_csv(f'{wpath}/spec_norm.csv',index=False)
        df_label.to_csv(f'{wpath}/spec_lbl.csv', index=False)
    return dfspec_norm,vmin,vrng,df_label

def load_norm_params(vpath):
    vmin=np.loadtxt(f'{vpath}/vmin.txt')
    vrng=np.loadtxt(f'{vpath}/vrng.txt')
    return vmin, vrng

def prepro_photos(PHOTO_DATA, vmin, vrng, base, ftr,w=False, wpath=None):
    dfphoto=pd.read_csv(PHOTO_DATA)
    assert len(ftr)==len(vmin) & len(ftr)==len(vrng)
    df=((dfphoto[ftr]-vmin)/vrng).clip(0,1)
    if w:
        print(f'writing photo_norm_{base} to {wpath}')
        df.to_csv(f'{wpath}/photo_norm_{base}.csv', index=False)
    return df




def prepro_photo_spec(PHOTO_DATA, SPEC_DATA, base,ftr, w=True, wpath=None):
    dffull=pd.read_csv(SPEC_DATA)
    dffull=dffull[dffull['class']!='GALAXY']
    print(dffull.shape)
    dfspec_norm,vmin,vrng,vmean,vstd,df_lbl=get_std_norm_pd(dffull[ftr],dffull[['class','subclass','lu']])
    # print(vmin,vrng,vmean,vstd)
    # print(dfspec_norm.all().isnull().sum())
    df_lbl=get_stellar_pd(df_lbl)  
    # dfspec_norm.to_csv(f'{wpath}/spec_norm.csv',index=False)
    print(dfspec_norm.shape)
    # df_lbl.to_csv(f'{wpath}/spec_lbl.csv', index=False)   
    dfphoto=pd.read_csv(PHOTO_DATA)
    dfphoto_norm=(dfphoto[ftr]-vmean)/vstd
    dfphoto_norm=(dfphoto_norm-vmin)/vrng
    dfphoto_norm=dfphoto_norm[(dfphoto_norm[dfphoto_norm.columns] >= 0).all(axis=1) & (dfphoto_norm[dfphoto_norm.columns] <= 1).all(axis=1)]
    assert (dfphoto_norm.min().min()>=0) & (dfphoto_norm.max().max()<=1)
    print(dfphoto_norm.shape)
        # df.to_csv(f'{wpath}/photo_norm_{base}.csv', index=False)
    return dfphoto_norm, dfspec_norm,df_lbl


    # if w:
    #     print(f'writing photo_norm_para_{base} to {wpath}')
    #     np.savetxt(f'{wpath}/vmin.txt',vmin)
    #     np.savetxt(f'{wpath}/vrng.txt',vrng)
    #     np.savetxt(f'{wpath}/vmean.txt',vmean)
    #     np.savetxt(f'{wpath}/vstd.txt',vstd)