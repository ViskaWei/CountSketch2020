import copy
import numpy as np
import pandas as pd

def get_std_ftr_pd(df_data0, vmean0, vstd0, ftr,sig):
    vmean, vstd=vmean0[ftr], vstd0[ftr]
    df_data=df_data0[ftr]
    df_norm=(df_data-vmean)/vstd
    print(df_norm.describe())
    df_sig=df_norm[(df_norm[df_norm.columns]>-sig).all(axis=1)&(df_norm[df_norm.columns]<sig).all(axis=1)]    
    print(df_sig.shape,df_norm.shape )
    df_snorm=(df_sig/(2*sig))+0.5    
    assert (df_snorm.min().min()>=0) & (df_snorm.max().max()<=1)
    return df_snorm
    
def get_std_pd(df_data, vmean, vstd, sig):
    # vmean, vtd=vmean0[ftr],vstd0[ftr]
    df_norm=(df_data-vmean)/vstd
    df_sig=df_norm[(df_norm[df_norm.columns]>-sig).all(axis=1)&(df_norm[df_norm.columns]<sig).all(axis=1)]    
    print(df_sig.shape,df_norm.shape )
    df_snorm=(df_sig/(2*sig))+0.5    
    assert (df_snorm.min().min()>=0) & (df_snorm.max().max()<=1)
    return df_snorm

def get_col_norm_pd(df,r=0.01,w=False,std=False): 
    df1=(df-df.mean())/df.std() if std else df
    vmin=df1.quantile(r)
    vrng=df1.quantile(1-r)-vmin
    df1=((df1-vmin)/vrng).clip(0,1)
    if w: return df1,vmin,vrng
    return df1

def get_minmax_pd(df,r=0.01, vmin=None, vmax=None): 
    if vmin is None: vmin=np.min(df.quantile(r))
    if vmax is None: vmax=np.max(df.quantile(1-r))
    print('full min max',vmin,vmax )
    df_norm=((df-vmin)/(vmax-vmin)).clip(0,1)
    return df_norm

def get_std_norm_pd(df,df_lbl=None): 
    vmean,vstd=df.mean(),df.std()
    df=(df-vmean)/vstd
    vmin=df.quantile(0.005)
    vrng=df.quantile(1-0.005)-vmin
    df=((df-vmin)/vrng)
    print(df.shape)
    df=df[(df[df.columns] >= 0).all(axis=1)&(df[df.columns] <= 1).all(axis=1)]
    assert (df.min().min()>=0) & (df.max().max()<=1)
    print(df.shape)
    if df_lbl is not None:
        df_lbl=df_lbl.iloc[df.index]
        assert len(df_lbl)==len(df)
        return df,vmin,vrng,vmean,vstd,df_lbl
    return df,vmin,vrng,vmean,vstd
    
def get_encode_stream(df_norm, base,dtype):
    mat=(df_norm*(base-1)).round()
    assert (mat.min().min()>=0) & (mat.max().max()<=base-1)
    mat_encode=horner_encode(mat,base,dtype) 
    mat_decode=horner_decode(mat_encode,base,len(mat.keys()),dtype)  
    assert (mat_decode.min().min()>=0) & (mat_decode.max().max()<=base-1)
    try:
        assert np.sum(abs(mat_decode-mat.values))<=0.0001    
    except:
        print(np.nonzero(np.sum(abs(mat_decode-mat.values),axis=1)), np.sum(abs(mat_decode-mat.values)))
        raise 'overflow, try lower base or fewer features'     
    return mat_encode

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