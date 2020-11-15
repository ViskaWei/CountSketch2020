import os
import numpy as np
import pandas as pd
import copy
# import umap
# from collections import Counter
import warnings
warnings.simplefilter("ignore")

t0_dict = dict({ 'W':'#003366',
                'K7': 'maroon',
                'A2':'#8FC3FD',
                  'F8':'#FF8E00',
                  'G8': 'red',   
                'M0':'pink',
                   'Q': '#556B2F',
                 'x':'lightgray'})
lu_dict = dict({'I':'#0D2B68',
                  'II':'#8FC3FD',
                  'III': 'red',
                  'IV': '#FF5003',
                   'V': '#FACF4A',
                  'W':'#556B2F'})

def get_subclass_pd(df_label,LBL_PATH='../data/sdss_stars/label.csv'):
    labeldict_pd=pd.read_csv(LBL_PATH)
    lu_dict={}
    lbl_dict={}
    t_dict={}
    for ii, subcls in enumerate(labeldict_pd['subclass'].values):
        lbl_dict[subcls]=labeldict_pd['T'][ii]
        lu_dict[subcls]=labeldict_pd['L'][ii]
        t_dict[subcls]=labeldict_pd['class5'][ii]   
    df_label['subclass'][df_label['subclass'].isnull()]='qN'                
    df_label['subclass'][df_label['subclass']=='']='qN'
    df_label['lbl']=df_label['subclass'].apply(lambda x: lbl_dict[x])
    df_label['lu']=df_label['subclass'].apply(lambda x: lu_dict[x].strip())
    df_label['class5']=df_label['subclass'].apply(lambda x: t_dict[x])
    print(df_label['class5'].unique())
    print(df_label['lu'].unique())
    df_label['t']=df_label['lbl'].apply(lambda x: str(x)[0])
    df_label['t8']=df_label['t']
    df_label['t8'][(df_label['t']=='L')|(df_label['t']=='T')]='M'
    df_label['t8'][(df_label['t']=='O')|(df_label['t']=='B')]='A'
#     df_label.to_csv('./data/sdss_stars/DR13v1/spec_lbl.csv',index=False)
    return df_label


# label2_keys=['O','B','A','F','G','K','M','W','C']
label2_keys=['A', 'B', 'C', 'V', 'F', 'G', 'K', 'L', 'M', 'O', 'T', 'W']
label3_keys=['OBA','FG','KM']
label2_keys2=['O','F','M']
# label4_keys=['OB','C','M']
# label4_keys=['OB', 'C1','V', 'M', 'WD']
label4_keys=['B', 'CB', 'WD']
# label4_keys=['C1','V']
# [print(ii, np.sum(df_label["label2"]==ii)) for ii in label2_keys]
def get_stellar_label(x,cls):
    if cls=='' or cls=='x' : return 'x'
    elif cls=='STAR':
        try:
            return get_star_label(x)
        except:
            return 'sx'
    elif cls=='QSO':
        try:
            return get_quasar_label(x)
        except:
            return 'qN'
    elif cls=='GALAXY': return 'glx'
    
def get_star_label(x):
    return "s"+x[0]
def get_quasar_label(x):
    if x=='' or x=='x' or x=='QN' or x=='BROADLINE': return 'qN'
    if x[-1]=='E':return 'qB'
    return 'qO'


def get_stellar_pd(df, lbl='label',islbl=True):
    if islbl: 
        assert sum(df['class']=='')==0
    else:
        df['subclass'][df['class'].isnull()]='x'
        df['class'][df['class'].isnull()]='x'
    df[lbl]=df.apply(lambda x: get_stellar_label(x["subclass"], x["class"]), axis=1)
    print(df[lbl].unique())
    print(df[df['class']=='QSO'][lbl].unique())
    print(df[df['class']=='STAR'][lbl].unique())
    get_QSG_subclass_label(df,lbl=lbl,lkey='l8',lkey2='l5')
    # df['subclass'][df['subclass'].isnull()]='qN'
    return df

def get_QSG_subclass_label(df,lbl='label',lkey='l8',lkey2='l5'):
    df[lkey]='sx'
    df[lkey][(df[lbl]=='sO')| (df[lbl]=='sB')| (df[lbl]=='sA')]='A'
    df[lkey][(df[lbl]=='sG')|(df[lbl]=='sF')]='F'
    df[lkey][(df[lbl]=='sK')]='K'
    df[lkey][(df[lbl]=='sM')| (df[lbl]=='sL')| (df[lbl]=='sT')]='M'
    df[lkey][(df[lbl]=='sW')]='W'
#     df[lkey][(df['subclass']=='Carbon')|(df['subclass']=='Carbon_lines') |(df['subclass']=='CarbonWD')]='C'
    df[lkey][(df[lbl]=='qN')]='Q'
    df[lkey][(df[lbl]=='qB')| (df[lbl]=='qO')]='qx'
    df[lkey][(df[lbl]=='glx')]='glx'
    df[lkey][(df[lbl]=='x')]='x' 
    print(df[lkey].unique())
    df[lkey2]='sx'
    df[lkey2][(df[lkey]=='W')]='W'
    df[lkey2][(df[lkey]=='Q')]='Q'
    df[lkey2][(df[lkey]=='F')| (df[lkey]=='G')| (df[lkey]=='A')]='A'
    df[lkey2][(df[lkey]=='K')|(df[lkey]=='M')]='K'
    df[lkey2][(df[lkey]=='qx')]='qx'
    df[lkey2][(df[lbl]=='glx')]='glx'
    df[lkey2][(df[lbl]=='x')]='x' 
    print(df[lkey2].unique())
    return None

