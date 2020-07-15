import sys
sys.path.insert(0, '/home/swei20/cancerHH/AceCanZ/code')
import os
import time
import copy
import joblib
import numpy as np
import pandas as pd
from sdss.prepro.dataset import prepro_std_specs,prepro_std_photos
from sdss.postpro.project import get_spec_mapping,get_mapping_pd,get_umap_pd,get_spec_label
from util.prepro import get_encode_stream
from util.HH import get_HH_pd
import getpass
from SciServer import Authentication, CasJobs
PRE_SPEC,PRE_PHOTO_HH,PRE_UMAP=1,1,1
MAP_SPEC,UPLOAD_SCI=1,0
EXACT=1
TEST=1
# ftr=['ug05','ur75','ui05','uz75','gr10','gi75','gz75','rz75','iz10','ps_u']
ftr=['ug', 'gr', 'ri', 'iz', 'ur', 'gi', 'rz', 'ui',
       'gz', 'uz', 'ug5', 'gr5', 'ri5', 'iz5', 'ur5', 'gi5', 'rz5', 'ui5',
       'gz5', 'uz5']
ftr_len=len(ftr)
ftr_str=[f'{ii}' for ii in range(ftr_len)]
base, topk, umap_comp=20,20000,4
dtype='int64'
name=f'c{ftr_len-1}_b{base}t{topk}u{umap_comp}u_ext'
PRETRAIN=f'./data/sdss_stars/pretrain/{name}'
if TEST:
    SPEC_DATA='./data/sdss_stars/DR13color/speccolor_test.csv'
    PHOTO_DATA='./data/sdss_stars/DR13color/photo_test.csv'
else:
    PHOTO_DATA='./data/sdss_stars/DR13/30M_photo_r.csv'
    SPEC_DATA='./data/sdss_stars/DR13color/specDR13.csv'

def main():
    try:
        os.mkdir(PRETRAIN)
    except:
        print('here we go!')
    if PRE_SPEC:
        dfspec, vmean,vstd, df_lbl= prepro_std_specs(SPEC_DATA, ftr=ftr, sig=3.0, w=True,wpath=PRETRAIN)
    elif PRE_PHOTO_HH:
        vmean=np.loadtxt(f'{PRETRAIN_PATH}/vmean.txt')
        vstd=np.loadtxt(f'{PRETRAIN_PATH}/vstd.txt')

    if PRE_PHOTO_HH:
        print('=====================PREPRO PHOTO====================')
        dfphoto=prepro_std_photos(PHOTO_DATA, vmean, vstd, ftr=ftr, sig=3.0)
        photo_stream=get_encode_stream(dfphoto,base,dtype)
        photo_HH=get_HH_pd(photo_stream,base,ftr_len,dtype, EXACT,topk,r=16, d=1000000,c=None,device=None)
        if not EXACT: 
            assert len(photo_HH)<=topk 
        else:
            photo_HH=photo_HH[:topk]
        photo_HH.to_csv(f'{PRETRAIN}/photo_HH.csv', index=False)
    elif PRE_UMAP:
        photo_HH=pd.read_csv(f'{PRETRAIN}/photo_HH.csv', columns=list(range(ftr_len)))
    
    if PRE_UMAP:
        print('=============GETTING UMAP============')
        try: photo_uT=get_umap_pd(photo_HH,list(range(ftr_len)), umap_comp)
        except: photo_uT=get_umap_pd(photo_HH,ftr_str, umap_comp)
        joblib.dump(photo_uT, f'{PRETRAIN}/photo_uT.sav')
        photo_HH.to_csv(f'{PRETRAIN}/photo_HH.csv', index=False)
    elif MAP_SPEC:
        photo_uT=joblib.load(f'pretrain/photo_uT.sav')
    
    if MAP_SPEC:
        if not PRE_SPEC:
            dfspec=pd.read_csv(f'{PRETRAIN}/spec_norm.csv')
        dfspec_block=(dfspec*(base-1)).round()
        assert (dfspec_block.min().min()>=0) & (dfspec_block.max().max()<=base-1)
        spec_pm=get_mapping_pd(dfspec_block,photo_uT,dfspec.keys())
        spec_pm.to_csv(f'{PRETRAIN}/spec_pm_e{EXACT}.csv',index=False)
    else:
        spec_pm=pd.read_csv(f'{PRETRAIN}/spec_pm_e{EXACT}.csv')

    spec_pmlbl= pd.concat([spec_pm,df_lbl],axis=1)
    spec_pmlbl.to_csv(f'{PRETRAIN}/spec_pm_e{EXACT}_lbl.csv',index=False)
    
    if UPLOAD_SCI:
        username = 'viskawei'
        password='Bstrong1178!'
    # password = getpass.getpass()
        sciserver_token = Authentication.login(username, password)
        CasJobs.uploadPandasDataFrameToTable(dataFrame=photo_HH, tableName=f'{name}b{base}e{EXACT}std', context="MyDB")

            

if __name__ == "__main__":
    main()
        
        
        
        
        
        
        
        
        
        