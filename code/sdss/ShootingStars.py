import sys
sys.path.insert(0, '/home/swei20/cancerHH/AceCanZ/code')
import os
import time
import copy
import joblib
import numpy as np
import pandas as pd
from sdss.prepro.dataset import prepro_specs,prepro_photos
from sdss.postpro.project import get_spec_mapping,get_mapping_pd,get_umap_pd,get_spec_label
from util.prepro import get_encode_stream
from util.HH import get_HH_pd

# os.environ["CUDA_VISIBLE_DEVICES"]="2"

# PHOTO_DATA='./data/sdss_stars/DR13/30M_photo_r.csv'
'''
PREPRO_NORM_PARAMS=0
PREPRO_NORM_PHOTO=1
PREPRO_STREAM_PHOTO=1
PREPRO_STREAM_SPEC=0
EXACT_COUNTING=1
PREPRO_HH_PHOTO=1
PREPRO_HH_SPEC=0
MAP_PHOTO=1
'''
'''
PREPRO_NORM_PARAMS=1
PREPRO_NORM_PHOTO=1
PREPRO_STREAM_PHOTO=1
PREPRO_STREAM_SPEC=1
EXACT_COUNTING=1
PREPRO_HH_PHOTO=1
PREPRO_HH_SPEC=1
MAP_PHOTO=0
'''
PREPRO_NORM_PARAMS,PREPRO_NORM_PHOTO,PREPRO_STREAM_PHOTO=0,0,0

PREPRO_STREAM_SPEC=0
EXACT_COUNTING=1
PREPRO_HH_PHOTO=0
PREPRO_HH_SPEC=0
MAP_PHOTO=0
# '''
PREPRO_SPEC_UT=1
PREPRO_SPEC_UMAP=0
PREPRO_PHOTO_UMAP=1
WRITING=1
TEST=0
# ftr=['ug','gr','ri','iz','r']
ftr=['ug', 'ur', 'ui', 'uz', 'gz', 'gi', 'gr', 'rz', 'ri', 'iz', 'z']
ftr_len=len(ftr)
ftr_str=[f'{ii}' for ii in range(ftr_len)]
base=40
dtype='int64'
topk=20000
umap_comp=3
name=f'c{ftr_len-1}_b{base}t{topk}r'
PRETRAIN_PATH=f'./data/sdss_stars/pretrain/{name}t{TEST}'

# if TEST:
#     SPEC_DATA='./data/sdss_stars/DR13v1/544k_spec_ztest.csv'
#     PHOTO_DATA='./data/sdss_stars/DR13/NIPS30Mtest.csv'
# else:
#     PHOTO_DATA='./data/sdss_stars/DR13v1/544k_spec_z.csv'
#     SPEC_DATA='./data/sdss_stars/DR13/544k_spec.csv'  

if TEST:
    SPEC_DATA='./data/sdss_stars/DR13/544k_spec_objidtest.csv'
    PHOTO_DATA='./data/sdss_stars/DR13/30M_photo_rtest.csv'
else:
    PHOTO_DATA='./data/sdss_stars/DR13/30M_photo_r.csv'
    SPEC_DATA='./data/sdss_stars/DR13/544k_spec_objid.csv'    

def main():
    try:
        os.mkdir(PRETRAIN_PATH)
    except:
        print('here we go!')

    if PREPRO_HH_PHOTO:
        if PREPRO_STREAM_PHOTO:
            if PREPRO_NORM_PHOTO:
                if PREPRO_NORM_PARAMS:
                    print('=====================PREPRO SPECS====================')
                    dfspec,vmin,vrng,df_lbl=prepro_specs(SPEC_DATA, ftr, r=0.01,w=True,wpath=PRETRAIN_PATH)
                elif PREPRO_NORM_PHOTO:
                    vmin=np.loadtxt(f'{PRETRAIN_PATH}/vmin.txt')
                    vrng=np.loadtxt(f'{PRETRAIN_PATH}/vrng.txt')
                print('=====================PREPRO PHOTO====================')
                dfphoto=prepro_photos(PHOTO_DATA, vmin, vrng, base,ftr, w=True, wpath=PRETRAIN_PATH)
            else:
                print('=====================LOADING PHOTO NORM ====================')
                dfphoto=pd.read_csv(f'{PRETRAIN_PATH}/photo_norm_{base}.csv',index=False)
            print('=====================ENCODE PHOTO ====================')
            photo_stream=get_encode_stream(dfphoto, base,dtype)
            np.savetxt(f'{PRETRAIN_PATH}/photo_stream.txt',photo_stream)
        else:
            print('=====================LOADING PHOTO STREAM ====================')
            photo_stream=np.loadtxt(f'{PRETRAIN_PATH}/photo_stream.txt')
        print('===================== COUNTING PHOTO HH==================')
        photoHH_pd=get_HH_pd(photo_stream,base,ftr_len,dtype, EXACT_COUNTING,topk,r=16, d=1000000,c=None,device=None)
        photoHH_pd.to_csv(f'{PRETRAIN_PATH}/photo_HH.csv', index=False)
    else:
        photoHH_pd=pd.read_csv(f'{PRETRAIN_PATH}/photo_HH.csv')
    print('photoHH_pd',photoHH_pd)

    if PREPRO_STREAM_SPEC:
        if not PREPRO_NORM_PARAMS:
            dfspec=pd.read_csv(f'{PRETRAIN_PATH}/spec_norm.csv')
            df_lbl=pd.read_csv(f'{PRETRAIN_PATH}/spec_lbl.csv')
        print('=====================ENCODING SPEC ====================')
        spec_stream=get_encode_stream(dfspec, base,dtype)
        np.savetxt(f'{PRETRAIN_PATH}/spec_stream.txt',spec_stream)
        df_lbl['encode']=spec_stream
        df_lbl.to_csv(f'{PRETRAIN_PATH}/spec_lbl_encode.csv', index=False)
    else:
        df_lbl=pd.read_csv(f'{PRETRAIN_PATH}/spec_lbl_encode.csv')
        if PREPRO_HH_SPEC:
            spec_stream=np.loadtxt(f'{PRETRAIN_PATH}/spec_stream.txt')
    

        
    if PREPRO_HH_SPEC:  
        print('=====================COUNTING PHOTO HH====================')
        specHH_pd=get_HH_pd(spec_stream,base,ftr_len,dtype,True,topk)
        print('=====================UMAPPING SPEC ====================')
        specHH_pd.to_csv(f'{PRETRAIN_PATH}/specHH_pd.csv',index=False)
    elif MAP_PHOTO:
        specHH_pd=pd.read_csv(f'{PRETRAIN_PATH}/specHH_pd.csv')
        print('specHH_pd',specHH_pd)
        
    if MAP_PHOTO:
        print('=============MAPPING PHOTO============')
        if PREPRO_SPEC_UMAP:
            HH_pdQS,umapT_spec= get_spec_mapping(specHH_pd,ftr, df_lbl, base,name,umap_comp,HH_cut=20000)
            print('HH_pdQS',HH_pdQS)
            # joblib.dump(model_dict, f'{PRETRAIN_PATH}/model_b{base}.sav')
            joblib.dump(umapT_spec, f'{PRETRAIN_PATH}/umap_spec_b{base}.sav')
            HH_pdQS.to_csv(f'{PRETRAIN_PATH}/spec_HHQS.csv',index=False)
        else:
            umapT_spec=joblib.load(f'{PRETRAIN_PATH}/umap_spec_b{base}.sav')
            # umapT=joblib.load(f'{PRETRAIN_PATH}/umap_b{base}.sav')
        print('=====================UMAP PROJECTING PHOTO ====================')
        photo_mapped=get_mapping_pd(photoHH_pd,umapT_spec, list(range(ftr_len)))
        print('=====================SAVING SMAPPED PHOTO ====================')
        photo_mapped.to_csv(f'{PRETRAIN_PATH}/photoUTe{EXACT_COUNTING}.csv',index=False)
    else:
        if not PREPRO_HH_SPEC :  df_lbl=pd.read_csv(f'{PRETRAIN_PATH}/spec_lbl_encode.csv')
        print('=============MAPPING SPEC============')
        if PREPRO_SPEC_UT:
            if PREPRO_PHOTO_UMAP:
                photoHH_pdh=photoHH_pd[:topk]
                print(photoHH_pdh)
                try: umapT_photo=get_umap_pd(photoHH_pdh,list(range(ftr_len)), umap_comp)
                except: umapT_photo=get_umap_pd(photoHH_pdh,ftr_str, umap_comp)
                joblib.dump(umapT_photo, f'{PRETRAIN_PATH}/umap_photo_b{base}.sav')
                photoHH_pdh.to_csv(f'{PRETRAIN_PATH}/photoHH_pdh.csv', index=False)
            else:
                umapT_photo=joblib.load(f'pretrain/umap_photo_b{base}.sav')
            if not PREPRO_NORM_PARAMS:
                dfspec=pd.read_csv(f'{PRETRAIN_PATH}/spec_norm.csv')
            dfspec=(dfspec*(base-1)).round()
            spec_pm=get_mapping_pd(dfspec,umapT_photo,dfspec.keys())
            spec_pm.to_csv(f'{PRETRAIN_PATH}/spec_pm_e{EXACT_COUNTING}.csv',index=False)
        else:
            spec_pm=pd.read_csv(f'{PRETRAIN_PATH}/spec_pm_e{EXACT_COUNTING}.csv')
        specUT_lbled= pd.concat([spec_pm,df_lbl],axis=1)
        specUT_lbled.to_csv(f'{PRETRAIN_PATH}/spec_pm_e{EXACT_COUNTING}_lbl.csv',index=False)
        

if __name__ == "__main__":
    main()
        
        
        
        
        
        
        
        
        
        