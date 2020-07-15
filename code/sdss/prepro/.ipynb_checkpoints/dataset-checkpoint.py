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
from code.prepro.label import get_stellar_pd

ckeys=['ug','ur', 'ui','uz','gz', 'gi', 'gr','rz','ri','iz']
mkeys=['z']
# SPEC_DATA='../sloan_stars/DR13v1/sql_544k_lbt.csv'
ftr=ckeys+mkeys
dtype='int64'
PRETRAIN_PATH='./pretrain'


def get_norm_pd(df,r=0.01,std=False): 
    df1=(df-df.mean())/df.std() if std else df
    vmin=df1.quantile(r)
    vrng=df1.quantile(1-r)-vmin
    df_new=((df1-vmin)/vrng).clip(0,1)
    return df_new,vmin,vrng

def get_prepro_pds(df_full,ftr=ftr, r=0.01):
    df_norm,vmin,vrng=get_norm_pd(df_full[ftr], r=r,std=False)
    df_label=get_stellar_pd(df_full[['class','subclass']])    
    return df_norm,vmin,vrng,df_label


def prepro_specs(SPEC_DATA,ftr=ftr, r=0.01,w=True,wpath=PRETRAIN_PATH):
    dfspec=pd.read_csv(SPEC_DATA)
    dfspec_norm,vmin,vrng,df_label=get_prepro_pds(dfspec,r=0.01)
    print(dfspec_norm.head(),vmin,vrng,)
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

def prepro_photos(PHOTO_DATA, vmin, vrng, base, w=False, ftr=ftr,vpath=PRETRAIN_PATH):
    dfphoto=pd.read_csv(PHOTO_DATA)
    assert len(ftr)==len(vmin) & len(ftr)==len(vrng)
    df=((dfphoto[ftr]-vmin)/vrng).clip(0,1)
    if w:
        print(f'writing photo_norm_{base} to {vpath}')
        df.to_csv(f'{vpath}/photo_norm_{base}.csv', index=False)
    return df

    # df_label['encode']=full_encode;

#######################ENCODE######################
def horner_encode(mat,base,dtype):
    r,c=mat.shape
    print('samples:',r,'ftrs:',c, 'base:',base)
    encode=np.zeros((r),dtype=dtype)
    for ii, key in enumerate(mat.keys()):
        val=(mat[key].values).astype(dtype)
        encode= encode + val*(base**ii)
#         print(ii,val, encode)
    return encode

def horner_decode(encode,base, dim,dtype):
    arr=copy.deepcopy(np.array(encode))
    decode=np.zeros((len(arr),dim), dtype=dtype)
    for ii in range(dim-1,-1,-1):
        digits=arr//(base**ii)
        decode[:,ii]=digits
        arr= arr% (base**ii)
#         print(digits,arr)
    return decode

def get_encode_stream1D(df_norm, base,dtype):
    mat=(df_norm*(base-1)).round()
    assert (mat.min().min()>=0) & (mat.max().max()<=base-1)
    mat_encode=horner_encode(mat,base,dtype) 
    mat_decode=horner_decode(mat_encode,base,len(mat.keys()),dtype)  
    try:
        assert np.sum(abs(mat_decode-mat.values))<=0.0001    
    except:
        print(np.nonzero(np.sum(abs(mat_decode-mat.values),axis=1)), np.sum(abs(mat_decode-mat.values)))
        raise 'overflow, try lower base or fewer features'     
    return mat_encode
