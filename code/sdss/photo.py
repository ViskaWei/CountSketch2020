import sys
sys.path.insert(0, '/home/swei20/cancerHH/AceCanZ/code')
import os
import time
import copy
import joblib
import numpy as np
import pandas as pd
from sdss.prepro.dataset import prepro_specs,prepro_photos,prepro_photo_spec
from sdss.postpro.project import get_spec_mapping,get_mapping_pd,get_umap_pd,get_spec_label
from util.prepro import get_encode_stream
from util.HH import get_HH_pd
import getpass
from SciServer import Authentication, CasJobs
# os.environ["CUDA_VISIBLE_DEVICES"]="2"
PRE_NORM,PRE_HH,PRE_UMAP=1,1,1
MAP_SPEC,UPLOAD_SCI=1,0
EXACT=0
TEST=0
# ftr=['ug','gr','ri','iz','r']
ftr=['ug', 'ur', 'ui', 'uz', 'gz', 'gi', 'gr', 'rz', 'ri', 'iz', 'r']
ftr_len=len(ftr)
ftr_str=[f'{ii}' for ii in range(ftr_len)]
base, topk, umap_comp=28,20000,4
dtype='int64'
name=f'c{ftr_len-1}_b{base}t{topk}u{umap_comp}r'
PRETRAIN=f'./data/sdss_stars/pretrain/{name}'
if TEST:
    SPEC_DATA='./data/sdss_stars/DR13/540k_spectest.csv'
    PHOTO_DATA='./data/sdss_stars/DR13/30M_photo_rtest.csv'
else:
    PHOTO_DATA='./data/sdss_stars/DR13/30M_photo_r.csv'
    SPEC_DATA='./data/sdss_stars/DR13/540k_spec.csv'    

def main():
    try:
        os.mkdir(PRETRAIN)
    except:
        print('here we go!')
    if PRE_NORM:
        dfphoto, dfspec,df_lbl=prepro_photo_spec(PHOTO_DATA, SPEC_DATA, base,ftr, wpath=PRETRAIN)
    if PRE_HH:
        print('=====================ENCODE PHOTO ====================')
        photo_stream=get_encode_stream(dfphoto,base,dtype)
        spec_stream=get_encode_stream(dfspec,base,dtype)
        # np.savetxt(f'{PRETRAIN}/photo_stream.txt',photo_stream)
        # np.savetxt(f'{PRETRAIN}/spec_stream.txt',spec_stream)
        df_lbl['encode']=spec_stream
        df_lbl.to_csv(f'{PRETRAIN}/spec_lbl_encode.csv', index=False)
        photo_HH=get_HH_pd(photo_stream,base,ftr_len,dtype, EXACT,topk,r=16, d=1000000,c=None,device=None)
        if not EXACT: 
            assert len(photo_HH)<=topk 
        else:
            photo_HH=photo_HH[:topk]
        photo_HH.to_csv(f'{PRETRAIN}/photo_HH.csv', index=False)
        spec_HH=get_HH_pd(spec_stream,base,ftr_len,dtype, True,topk)
        spec_HH.to_csv(f'{PRETRAIN}/spec_HH.csv', index=False)
    elif PRE_UMAP or MAP_SPEC:
        photo_HH=pd.read_csv(f'{PRETRAIN}/photo_HH.csv')
        spec_HH=pd.read_csv(f'{PRETRAIN}/spec_HH.csv')
        df_lbl=pd.read_csv(f'{PRETRAIN}/spec_lbl_encode.csv')
    print('photo_HH',photo_HH)
    print('spec_HH',spec_HH)
        
    if PRE_UMAP:
        print('=============GETTING UMAP============')
        try: photo_uT=get_umap_pd(photo_HH,list(range(ftr_len)), umap_comp)
        except: photo_uT=get_umap_pd(photo_HH,ftr_str, umap_comp)
        joblib.dump(photo_uT, f'{PRETRAIN}/photo_uT_b{base}.sav')
        photo_HH.to_csv(f'{PRETRAIN}/photo_HH.csv', index=False)
    elif MAP_SPEC:
        photo_uT=joblib.load(f'pretrain/photo_uT_b{base}.sav')
    
    if MAP_SPEC:
        if not PRE_NORM:
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
        
        
        
        
        
        
        
        
        
        