import sys
sys.path.insert(0,'/home/swei20/cancerHH/AceCanZ/code/')
import os
import numpy as np
import pandas as pd
import umap
import joblib
import getpass
from SciServer import Authentication, CasJobs
from collections import Counter
from cancer.prepro.encode import process_dataset_pc,process_pca,process_intensity,process_rebin
from cancer.postpro.project import get_umap_pd,get_kmean_lbl,get_pred_stream
from util.HH import get_HH_pd
from util.prepro import get_col_norm_pd, get_encode_stream
from sklearn import preprocessing
import warnings
import torch

np.random.seed(1178)
warnings.simplefilter("ignore")
data_dir = r'./data/bki'  
base=22
pca_comp=8
topk=20000
SMTH=1
dtype='uint64'
TEST = False
ISSMTH=True
EXACT=0
PREPRO_CUTOFF, PREPRO_NORM, PREPRO_STREAM=1,1,1
# PREPRO_HH,PREPRO_UMAP,PREPRO_KMEAN=1,1,1
# SAVE_ALL,PREDICT_ALL=1,1
# PREPRO_CUTOFF, PREPRO_NORM, PREPRO_STREAM=0,0,0
PREPRO_HH,PREPRO_UMAP,PREPRO_KMEAN=0,0,1
SAVE_ALL,PREDICT_ALL=0,0
UPLOAD_SCI=0

pckeys=[f'pc{ii}' for ii in range(pca_comp)]
nkeys=[f'n{ii}' for ii in range(pca_comp)]
N_cluster=10
num, pidx = 40 , 26
ONPCA, ONINT=0,0
ftr_len=8
name="n8"
PRETRAIN=f'./data/cancer_data/pretrain/RF/{name}1'
def main():
    try: 
        os.mkdir(PRETRAIN)
    except:
        'lets GO'
    print(PREPRO_CUTOFF, PREPRO_NORM, PREPRO_STREAM,PREPRO_HH,PREPRO_UMAP,PREPRO_KMEAN,SAVE_ALL,PREDICT_ALL,UPLOAD_SCI)
    if PREPRO_NORM:
        print(f'=================LOADING N={num} Smoothing {ISSMTH} =================')
        data1Ds, pc = process_dataset_pc(data_dir, num, pca_comp, ISSMTH, SMTH, TEST)
        intensity, pca_results=process_pca(data1Ds, pc, num)
        # df_pca=pd.DataFrame(pca_results, columns=list(range(pca_comp)))
        df_norm, mask,ftr_len0=process_intensity(pca_results, intensity, pca_comp,PREPRO_CUTOFF,ONPCA,ONINT,r=0.01,wdir=PRETRAIN)
        assert ftr_len0==ftr_len
        mask2d=mask.reshape((num,1004*1344))
        if SAVE_ALL:
            np.savetxt(f'{PRETRAIN}/mask_all.txt' , mask)
        else:
            mask0= mask2d[pidx]
            idxii=int(mask2d[:pidx].sum())
            idxjj=int(mask2d[:(pidx+1)].sum())
            assert idxjj-idxii == mask0.sum()
            print(mask0.shape, mask.sum(), 'saving mask')
            np.savetxt(f'{PRETRAIN}/mask{pidx}.txt' , mask0)
        # df_norm.to_csv(f'{PRETRAIN}/df_norm.csv',index=False)
        # df_normt=df_norm[idxii:idxjj]
#         df_normt.to_csv(f'{PRETRAIN}/df_norm{pidx}.csv',index=False)
    # elif PREPRO_STREAM:
    #     print(f'=================LOADING df_norm =================')
    #     df_norm=pd.read_csv(f'{PRETRAIN}/df_norm.csv')  

    if PREPRO_STREAM:
        print(f'=================ENCODING Base={base} =================')
        stream=process_rebin(df_norm, base, dtype)
        if SAVE_ALL:
            np.savetxt(f'{PRETRAIN}/stream_b{base}.txt' , stream)
        else:
            stream0=stream[idxii:idxjj]
            np.savetxt(f'{PRETRAIN}/stream_b{base}{pidx}.txt' ,stream0 )
    elif PREPRO_HH:
        print(f'=================LOADING STREAM =================')
        stream=np.loadtxt(f'{PRETRAIN}/stream_b{base}.txt')
        if not PREDICT_ALL:
            stream0=np.loadtxt(f'{PRETRAIN}/stream_b{base}{pidx}.txt')
    if PREPRO_HH:
        assert EXACT ==0
        topk=20000
        print(f'=================DECODE {ftr_len} DIM =================')
        HH_pd=get_HH_pd(stream,base,ftr_len, dtype, EXACT, topk, r=16, d=1000000,c=None,device=None)
        HH_pd.to_csv(f'{PRETRAIN}/HH_pd_b{base}e{EXACT}.csv',index=False)
    elif PREPRO_UMAP:
        print(f'=================LOADING HH_pd==============')
        HH_pd=pd.read_csv(f'{PRETRAIN}/HH_pd_b{base}e{EXACT}.csv')
        print(HH_pd.head()) 

    if PREPRO_UMAP:
        print(f'=================GETTING UMAP =================')
        # # lb,ub=int(HH_pd['freq'][0]*lbr),int(HH_pd['freq'][0])
        # HH_pdc=HH_pd[HH_pd['freq']>lb]
        # # print(len(HH_pdc),len(HH_pd),HH_pd['freq'][0],'lb',lb,'HHratio',lbr)
        # if len(HH_pdc)>20000: 
        HH_pdc=HH_pd[:20000]
        print(len(HH_pdc),len(HH_pd),HH_pd['freq'][0])
        print(f'=================LOADING HH_pd==============')
        umapT=get_umap_pd(HH_pdc,list(range(ftr_len)))
        # print(HH_pdc.keys())
        HH_pdc.to_csv(f'{PRETRAIN}/HH_pdh_b{base}e{EXACT}.csv',index=False)
    elif PREPRO_KMEAN:
        HH_pdc=pd.read_csv(f'{PRETRAIN}/HH_pdh_b{base}e{EXACT}.csv')

    if PREPRO_KMEAN:
        print(f'=================KMEAN CLUSTERING =================')
        kmap=get_kmean_lbl(HH_pdc, N_cluster, u1 = 'u1', u2 = 'u2')
        joblib.dump(kmap, f'{PRETRAIN}/kmap_k{N_cluster}e{EXACT}.sav')
        HH_pdc.to_csv(f'{PRETRAIN}/HH_pdh_b{base}e{EXACT}.csv',index=False)
    else:
        HH_pdc=pd.read_csv(f'{PRETRAIN}/HH_pdh_b{base}e{EXACT}.csv')
    
    if PREDICT_ALL:
        print(f'=================PREDICTING ALL {num} LABEL==============')
        if not PREPRO_NORM: mask=np.loadtxt(f'{PRETRAIN}/mask_all.txt')
        if not PREPRO_HH: stream=np.loadtxt(f'{PRETRAIN}/stream_b{base}.txt')
        pred_k=get_pred_stream(stream, mask, HH_pdc, f'k{N_cluster}' , val = 'HH', bg = 0, color = 0, sgn = 1 )
        pred_k=pred_k.reshape((num,1004,1344))
        print(f'=================SAVING PREDICTION of ALL {num} LABEL==============')
        np.savetxt(f'{PRETRAIN}/pred_k{N_cluster}e{EXACT}.txt', pred_k)
    else:
        print(f'=================PREDICTING id{pidx} LABEL==============')
        if not PREPRO_NORM: mask0=np.loadtxt(f'{PRETRAIN}/mask{pidx}.txt')
        if not PREPRO_HH: stream0=np.loadtxt(f'{PRETRAIN}/stream_b{base}{pidx}.txt')
        pred_k=get_pred_stream(stream0, mask0, HH_pdc, f'k{N_cluster}' , val = 'HH', bg = 0, color = 0, sgn = 1 )
        pred_k=pred_k.reshape((1004,1344))
        print(f'=================SAVING PREDICTION of id{pidx} LABEL==============')
        np.savetxt(f'{PRETRAIN}/pred_k{N_cluster}{pidx}_f{name}b{base}sm1c3sige{EXACT}.txt', pred_k)
    
    if UPLOAD_SCI:
        username = 'viskawei'
        password='Bstrong1178!'
    # password = getpass.getpass()
        sciserver_token = Authentication.login(username, password)
        CasJobs.uploadPandasDataFrameToTable(dataFrame=HH_pdc, tableName=f'b{base}sm{SMTH}f{name}sig3e{EXACT}_v1', context="MyDB")
    
if __name__ == "__main__":
    main()