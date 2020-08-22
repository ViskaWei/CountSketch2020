import os
import numpy as np
import pandas as pd
import copy
import torch
import time
import logging
from collections import Counter
import warnings
warnings.simplefilter("ignore")
from util.prepro import horner_decode
from util.csvec import CSVec

def get_exact_HH(stream,topk):
    print(f'=============exact counting HHs==============')
    t0=time.time()
    exactHH=np.array(Counter(stream).most_common())
    t=time.time()-t0
    print('exact counting time:{:.2f}'.format(t))
    return exactHH[:,0], exactHH[:,1], t

def get_HH_pd(stream,base,ftr_len, dtype, exact, topk, r=16, d=1000000,c=None,device=None):
    if exact:
        HH,freq,t=get_exact_HH(stream,topk)
        HHfreq=np.vstack((HH,freq))
    else:
        HH,freq,t=get_CS_HH(stream,d,c,r,topk,device)
        HHfreq=np.vstack((HH,freq))
    mat_decode_HH=horner_decode(HH,base,ftr_len, dtype)
    assert (mat_decode_HH.min().min()>=0) & (mat_decode_HH.max().max()<=base-1)
    HH_pd=pd.DataFrame(np.hstack((mat_decode_HH,HHfreq.T)), columns=list(range(ftr_len))+['HH','freq']) 
    HH_pd['rk']=HH_pd['freq'].cumsum()
    HH_pd['ra']=HH_pd['rk']/HH_pd['rk'].values[-1]
    # HH_dict={HH_pd['HH'].values[ii].astype('int'):ii for ii in range(len(HH_pd['HH'])) }
    return HH_pd

def get_CS_HH(stream,d,c,r,k,device):
    if c is None: c=10*k 
    stream_tr=torch.tensor(stream, dtype=torch.int64)
    csv = CSVec( d, c, r, k, device=device)
    t0=time.time()
    for ii in range(stream_tr.shape[0]//d+1):
        substream=stream_tr[ii*d:(ii+1)*d]
        csv.accumulateVec(substream)
    HHs=stream_tr.unique()
    tfreqs=csv.query(HHs).cpu().numpy()
    tfreqs=np.clip(tfreqs,0,None)
    idx=np.argsort(-1*tfreqs)    
    HHfreq=np.vstack((HHs.numpy(),tfreqs))
    t=time.time()-t0
    print('sketch counting time:{:.2f}'.format(t))
    shh,sfreq=HHfreq[:,idx][:,:k]
    return shh,sfreq,t