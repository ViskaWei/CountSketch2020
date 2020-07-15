import getpass
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import copy
import umap
import os
from collections import Counter
import seaborn as sns
from SciServer import Authentication, CasJobs
from pylab import rcParams
from sklearn.manifold import TSNE
from sklearn import preprocessing
import warnings
import time

warnings.simplefilter("ignore")

model_dict={}
rcParams['figure.figsize']=(20,4)
ckeys=['ug','ur', 'ui','uz','gz', 'gi', 'gr','rz','ri','iz']
mkeys=['z']

def get_prepro_pds(df_full,ftr, r=0.01):
    df_norm,vmin,vrng=get_norm_pd(df_full[ftr], r=r)
    df_label=get_stellar_pd(df_full[['class','subclass']])    
    return df_norm,vmin,vrng,df_label

def get_test_pd(df,ftr,vmin,vrng, umapT,base): 
#     df1=(df-df.mean())/df.std() if std else df
    assert len(ftr)==len(vmin) & len(ftr)==len(vrng)
    df_new=((df[ftr]-vmin)/vrng).clip(0,1)*base
    u_da=umapT.transform(df_new)   
    df['u1']=u_da[:,0]
    df['u2']=u_da[:,1]
    return df

def get_norm_pd(df,r=0.01,std=True): 
    df1=(df-df.mean())/df.std() if std else df
    vmin=df1.quantile(r)
    vrng=df1.quantile(1-r)-vmin
    df_new=((df1-vmin)/vrng).clip(0,1)
    return df_new,vmin,vrng

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

def get_stellar_pd(df, lbl='label'):
    df['subclass'][df['class'].isnull()]='x'
    df['class'][df['class'].isnull()]='x'
    df[lbl]=df.apply(lambda x: get_stellar_label(x["subclass"], x["class"]), axis=1)
    print(df[lbl].unique())
    print(df[df['class']=='QSO'][lbl].unique())
    print(df[df['class']=='STAR'][lbl].unique())
    get_QSG_subclass_label(df,lbl=lbl,lkey='l8',lkey2='l5')
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

#################################################################################### 


###############################################################
def get_HH_label(HH_pd,label_pd,encode_str='encode',label_str='label',HH_str='HH'):
    labels=[]
    ulabels=[]
    for ii, HH in enumerate(HH_pd[HH_str]):
        label=label_pd[label_pd[encode_str]==int(HH)][label_str].unique()
        lblstr="".join(label.astype(str))
        labels.append(lblstr)
        if len(label)>1:
            ulabels.append('mixed')
        else:
            ulabels.append(lblstr)
    HH_pd[label_str]=labels
    HH_pd[f'{label_str}U']=ulabels
    return HH_pd

###############################################################

def get_QS_stats(HH_pd,model_dict,name='m2c4p2',pd_key='classU',clusters_str=['QSO','STAR','mixed']):
    HH_pdh=HH_pd[(HH_pd['class']=='STAR')|(HH_pd['class']=='QSO')|(HH_pd['class']=='QSOSTAR')]
    HH_pdQS=HH_pdh[(HH_pdh['classU']!='mixed')]
    N_Q=(HH_pdh[pd_key]==clusters_str[0]).sum()
    N_S=(HH_pdh[pd_key]==clusters_str[1]).sum()
#     N_G=(HH_pdh[pd_key]==clusters_str[2]).sum()
    N_M=(HH_pdh[pd_key]==clusters_str[2]).sum()
    N_QS, N_A=len(HH_pdQS), len(HH_pdh)
    assert (abs(N_QS-N_Q-N_S)<1e-5) and( abs(N_M+N_QS-N_A)<1e-5)
    e, p_Q, p_S= N_QS/N_A, N_Q/(N_Q+N_M), N_S/(N_S+N_M)
    print(f'model:{name} e:{np.round(e*100,1)}%, Q purity: {np.round(p_Q*100,1)}%, S purity: {np.round(p_S*100,1)}%')
    QS_stats=[e,p_Q,p_S]
    model_dict[name]={"QS":QS_stats}
    return HH_pdQS,QS_stats


def get_label_pd(df,key,keys,intlbl=False):
    count=Counter(df[key])
    label_sorted=sorted(count,key=count.get,reverse=True)    
    label_dict={}
    for ii, label in enumerate(label_sorted):
        label_dict[label]=[ii,count[label]]
    df_label=df[keys]
    if intlbl:
        df_label['intlbl']=df[key].apply(lambda x: label_dict[x][0]).values
    return df_label,label_dict,label_sorted


def get_blockID(df, base):
    return (df*base).round()

def horner_encode(mat,base,dtype):
    r,c=mat.shape
    print(r,c, 'base:',base)
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


def get_encode_pd(ftr_pd,ftr, base,dtype):
    mat=ftr_pd[ftr]
    mat_encode=horner_encode(mat,base,dtype) 
    ftr_pd.loc[:,('encode')]=mat_encode   
    mat_decode=horner_decode(mat_encode,base,len(mat.keys()),dtype)  
    try:
        assert np.sum(abs(mat_decode-mat.values))<=0.0001    
    except:
        print(np.nonzero(np.sum(abs(mat_decode-mat.values),axis=1)), np.sum(abs(mat_decode-mat.values)))
        assert False     
    return mat_encode

def get_umap_pd(ftr_pd, ftr):
    umapT = umap.UMAP(n_components=2)
    umap_result = umapT.fit_transform(ftr_pd[ftr].values)
    ftr_pd['u1'] = umap_result[:,0]
    ftr_pd['u2'] = umap_result[:,1]
    return umapT

def get_tsne_pd(ftr_pd, ftr):
    tsneT = TSNE(n_components=2)
    X_embedded = tsneT.fit_transform(ftr_pd[ftr].values)
    ftr_pd['t1'] = X_embedded[:,0]
    ftr_pd['t2'] = X_embedded[:,1]
    return tsneT


def get_HH_pd(stream,base,ftr_len,dtype):
    count=Counter(stream)
    HH=sorted(count,key=count.get,reverse=True)
    mat_decode_HH=horner_decode(HH,base,ftr_len, dtype)
    mat_sorted=[count[HH[ii]] for ii in range(len(HH))]
    HH_pd=pd.DataFrame(np.vstack((mat_decode_HH.T,HH,mat_sorted)).T, columns=list(range(ftr_len))+['HH','freq']) 
    HH_dict={HH_pd['HH'].values[ii].astype('int'):ii for ii in range(len(HH_pd['HH'])) }
    return HH_pd,HH_dict


def get_model_dict(df_norm, ftr, df_label, base,lbc=8,dtype='int64' ):
    ftr_len=len(ftr)
    name=f"m{len(mkeys)}c{len(ckeys)}b{base}H{lbc}"
    print(ftr,ftr_len, name, end=' ')
    df_block = get_blockID(df_norm[ftr], base-1).astype(dtype)
    assert (df_block.min().min()>=0) & (df_block.max().max()<=base-1)
    full_encode=get_encode_pd(df_block,ftr,base,dtype);
    df_label['encode']=full_encode;
    HH_pd,HH_dict=get_HH_pd(full_encode,base,ftr_len,dtype);
    lb,ub=int(HH_pd['freq'][0]*0.01),int(HH_pd['freq'][0])
    lbc=np.min([lbc,ub])
    print('lbc',lbc,'lb',lb)
    HH_pdc=HH_pd[HH_pd['freq']>lbc]
    print(len(HH_pdc),len(HH_pd),HH_pd['freq'][0],  lb,lbc)
    get_HH_label(HH_pdc,df_label,encode_str='encode',label_str='class',HH_str='HH') 
#     return HH_pdc
    HH_pdQS,QS_stats=get_QS_stats(HH_pdc,model_dict,name=name,pd_key='classU',clusters_str=['QSO','STAR','mixed']) 
    umapT=get_umap_pd(HH_pdQS, list(range(len(ftr))));
#     sns.scatterplot('u1','u2',data=HH_pdQS)
    get_HH_label(HH_pdQS,df_label,encode_str='encode',label_str='l5',HH_str='HH')    
    HH_pd5=HH_pdQS[HH_pdQS[f'l5U']!='mixed']
    stat25=(len(HH_pd5),len(HH_pdQS), len(HH_pd5)/len(HH_pdQS))
    model_dict[name]['s25']=stat25
    print(stat25)
    get_HH_label(HH_pdQS,df_label,encode_str='encode',label_str='l8',HH_str='HH')    
    HH_pd8=HH_pdQS[HH_pdQS[f'l8U']!='mixed']
    stat58=[len(HH_pd8),len(HH_pdQS), len(HH_pd8)/len(HH_pdQS)]
    model_dict[name]['s58']=stat58
    print(stat58)
    f,axs=plt.subplots(1,3,figsize=(20,8))
    hues=['classU','l5U','l8U']
    for ii, ax in enumerate(axs.flatten()):
        sns.scatterplot('u1','u2',data=HH_pdQS,hue=hues[ii],ax=ax)
        if ii==0: ax.set_title(f'{name}, base{base}', fontsize=16)
        if ii==1: ax.set_title(f'{ftr}', fontsize=16)
    return HH_pdQS,umapT