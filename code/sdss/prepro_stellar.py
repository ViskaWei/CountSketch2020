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
import warnings
warnings.simplefilter("ignore")

model_dict={}
label_key='label'
rcParams['figure.figsize']=(20,4)
keys=["u","g","r","i","z"]
ckeys=['ug', 'gr', 'ri', 'iz']
magkeys=['r']
dmagkeys=[f'c_{x}' for x in ["u","g","r","i","z"]]

from sklearn import preprocessing

def get_quasar_label2(x):
    if x=='QN':
        return 'QN'
    if x[-1]=='E':
        if x[0]=='B':
            return 'QB'
        else:
            return 'QBO'
    return 'QO'

def get_star_label(x):
    return x[0]
def get_quasar_label(x):
    if x=='' or x=='QN' or x=='BROADLINE':
        return 'QN'
    if x[-1]=='E':
        return 'QB'
    return 'QO'
def get_stellar_label(x, y):
    if y=='':
        return 'unk'
    elif y=='STAR':
        return get_star_label(x)
    elif y=='QSO':
        return get_quasar_label(x)

def get_stellar_pd(df, label_key='label',diff=True):
    for x in ["u","g","r","i","z"]:
        if diff:
            df[f'c_{x}']=df[f'{x}']-df[f'ext_{x}']-df[f'der_{x}']
            keys=["u","g","r","i","z"]+[f'c_{x}' for x in ["u","g","r","i","z"]]
        else:
            df[f'mod_{x}']=df[f'ext_{x}']+df[f'der_{x}']
            keys=["u","g","r","i","z"]+[f'mod_{x}' for x in ["u","g","r","i","z"]]
#     df['subclass'][df['subclass']=='']='QN'
    df[label_key]=df.apply(lambda x: get_stellar_label(x["subclass"], x["class"]), axis=1)
    df1=df[keys+['class','subclass',label_key,'prob']]
    get_8cluster_label(df1,lkey='l8',lkey2='l5')
    return df1
    
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

def get_QScluster_label(df,lkey='l8',lkey2='l5'):
    df[lkey]='x'
    df[lkey][(df['label']=='O')| (df['label']=='B')| (df['label']=='A')]='A'
    df[lkey][(df['label']=='G')|(df['label']=='F')]='F'
    df[lkey][(df['label']=='K')]='K'
    df[lkey][(df['label']=='M')| (df['label']=='L')| (df['label']=='T')]='M'
    df[lkey][(df['label']=='W')]='W'
#     df[lkey][(df['subclass']=='Carbon')|(df['subclass']=='Carbon_lines') |(df['subclass']=='CarbonWD')]='C'
    df[lkey][(df['label']=='QN')]='Q'
    df[lkey][(df['label']=='QB')| (df['label']=='QO')]='Qx'
    print(df[lkey].unique())
    df[lkey2]='x'
    df[lkey2][(df['label']=='W')]='W'
    df[lkey2][(df['label']=='QN')]='Q'
    df[lkey2][(df[lkey]=='F')| (df[lkey]=='G')| (df[lkey]=='A')]='A'
    df[lkey2][(df[lkey]=='K')|(df[lkey]=='M')]='K'
    df[lkey2][(df[lkey]=='Qx')]='Qx'
    print(df[lkey2].unique())
    return None

def get_QS_stats(HH_pdh,model_dict,name='m2c4p2',pd_key='classU',clusters_str=['QSO','STAR','mixed']):
    HH_pdQS=HH_pdh[HH_pdh['classU']!='mixed']
    N_Q=(HH_pdh[pd_key]==clusters_str[0]).sum()
    N_S=(HH_pdh[pd_key]==clusters_str[1]).sum()
    N_M=(HH_pdh[pd_key]==clusters_str[2]).sum()
    N_QS, N_A=len(HH_pdQS), len(HH_pdh)
    assert (abs(N_QS-N_Q-N_S)<1e-5) and( abs(N_M+N_QS-N_A)<1e-5)
    e, p_Q, p_S= N_QS/N_A, N_Q/(N_Q+N_M), N_S/(N_S+N_M)
    print(f'model:{name} e:{np.round(e*100,1)}%, Q purity: {np.round(p_Q*100,1)}%, S purity: {np.round(p_S*100,1)}%')
    QS_stats=[e,p_Q,p_S]
    model_dict[name]=QS_stats
    return HH_pdQS,QS_stats

def get_ftr_pd(df,keys,ckeys, magkeys, dmagkeys, r1=0.02,r2=0.98):
    df_mag=df[keys[:5]]
    if magkeys is None:
        df_mag_norm=get_norm_pd(df_mag, vmin=15,vmax=22, method='minmax')
    elif magkeys==['r']:
        df_mag_norm=get_norm_pd(df_mag[['r']], vmin=15,vmax=20, method='minmax')
    else:
        df_mag_norm=get_norm_pd(df_mag[magkeys], vmin=15,vmax=22, method='minmax')
    df_magd_norm=get_norm_pd(df[dmagkeys], method='ratio',r1=r1,r2=r2)
#     df_color=get_color10(df_mag, keys=['u', 'g', 'r', 'i', 'z'])
    df_color=get_color_ckeys(df_mag, ckeys)
    df_color_norm=get_norm_pd(df_color,r1=r1,r2=r2, method='ratio')
    return df_mag_norm,df_magd_norm,df_color_norm

def get_color_ckeys(df_mag, ckeys):
    for key in ckeys:
        k1,k2=key[0],key[1]
        df_mag[f'{k1}{k2}']=df_mag[k1]-df_mag[k2]
#     return df_new
    return df_mag[ckeys]

def get_color10(df_mag, keys=['u', 'g', 'r', 'i', 'z']):
    l=len(keys)
    for ii,k1 in enumerate(keys):
        for jj in range(ii+1,l):
            k2=keys[jj]
            df_mag[f'{k1}{k2}']=df_mag[k1]-df_mag[k2]
    return df_mag.drop(columns=keys)

def get_10color_pd(df_mag):
    mag=df_mag.to_numpy()
    mag2=np.hstack((mag,mag)).T  
    mat_u,mat_g,mat_r=mag2[:5],mag2[1:6],mag2[2:7]
    d=np.vstack((mat_u-mat_g, mat_u-mat_r))
    d[4]=d[4]*(-1)
    d[-1]=d[-1]*(-1)
    d[-2]=d[-2]*(-1)
    df_color10=pd.DataFrame(d.T,columns=['ug','gr','ri','iz','zu','ur','gi','rz','ui','gz'])
    return df_color10

def run_step_pca(mag_data,pca_comp,base):
    mul_comb=mag_data.T.dot(mag_data)
    pc=run_step_pc(mul_comb, pca_comp)
    mag_pc=mag_data.dot(pc)
    intensity = (np.sum(mag_pc**2, axis = 1))**0.5
    mag_norm=np.divide(mag_pc,intensity[:,None])
    mag_rebin = np.trunc((mag_norm + 1) * base/2)
    return intensity,mag_norm,mag_rebin

def run_step_pc(mul_comb, pca_comp):
    # use svd since its commputational faster
    print("=============== run step PCA ===============")
    u,s,v = np.linalg.svd(mul_comb)
    assert np.allclose(u, v.T)
    print('Explained Variance Ratio', np.round(s/sum(s),3))
    pc = u[:,:pca_comp]
    return pc

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


def get_star_class(label_pd, label_pd_key, label_str_list, data_pd):
    star_dict={}
    for label in label_str_list:
        star_dict[label]=data_pd.loc[label_pd[label_pd_key]==label]
    return star_dict

def get_class_hist(para,star_dict,label_keys,base,ax=None ):
    if ax is None: ax=plt.gca()
    _=ax.hist([star_dict[label_key][para] for label_key in label_keys],log=True,bins=base, label=label_keys, density=True)
    
def get_class_hists(paras,label_keys,star_dict,base):
    l=len(paras)
    f,axs=plt.subplots(l,1, figsize=(20,3*l))    
    for ii,ax in enumerate(axs):
        get_class_hist(paras[ii], star_dict,label_keys,base,ax=ax)  
        ax.set_ylabel(paras[ii])
    handles, labels = ax.get_legend_handles_labels()
    f.legend(handles, labels, loc='center',ncol=len(label_keys))
    return None

def get_color_pd_old(df_mag,r1=None,r2=None,method='ratio',vmin=None,vmax=None):
    mag=df_mag.to_numpy()
    color=mag[:,:-1]-mag[:,1:]
    df_new=pd.DataFrame(color,columns=['ug','gr','ri','iz'])
    df_color_norm=get_norm_pd(df_new,vmin=vmin,vmax=vmax, method=method,r1=r1,r2=r2)
    return df_color_norm


def get_color_pd_old2(df_mag):
    mag=df_mag.to_numpy()
    color=mag[:,:-1]-mag[:,1:]
    df_new=pd.DataFrame(color,columns=['ug','gr','ri','iz'])
    df_new['ug']=df_new['ug'].clip(-0.25,1)  
    df_new['gr']=df_new['gr'].clip(-0.25,0.75)   
    df_new['ri']=df_new['ri'].clip(-0.3,0.5)    
    df_new['iz']=df_new['iz'].clip(-0.3,0.5)    
    df_new=(df_new-df_new.min())/(df_new.max()-df_new.min())
    return df_new

def get_10color_pd(df_mag):
    mag=df_mag.to_numpy()
    mag2=np.hstack((mag,mag)).T  
    mat_u,mat_g,mat_r=mag2[:5],mag2[1:6],mag2[2:7]
    d=np.vstack((mat_u-mat_g, mat_u-mat_r))
    d[4]=d[4]*(-1.0)
    d=d[:8]
    df_color10=pd.DataFrame(d.T,columns=['ug','gr','ri','iz','zu','ur','gi','rz'])
    return df_color10

def get_norm_pd(df_data, vmin=15,vmax=22, method='ratio',r1=0.05,r2=0.95): 
    if method=='ratio':
        vmin,vmax=np.mean(df_data.quantile(r1)),np.mean(df_data.quantile(r2))
    print(vmin,vmax)
    df_data=df_data.clip(vmin,vmax,axis=1)
    df_norm=(df_data-vmin)/(vmax-vmin)
    return df_norm

def get_isochrone(df_mag,vmin=-0.2,vmax=1):
    mag=df_mag.to_numpy()
    color=mag[:,:-1]-mag[:,1:]
    df_new=pd.DataFrame(color,columns=colorkeys)
    df_new['u-i']=mag[:,0]-mag[:,-1]
    df_new=df_new.clip(vmin,vmax)
    df_new=(df_new-vmin)/(vmax-vmin)
    return df_new

def get_blockID(df, base):
    return (df*base).round().astype('int')

def horner_encode(mat,base):
    r,c=np.shape(mat)
    encode=np.zeros((r))
    for col in range(c):
        encode+=mat[:,col]*base**col
#     encode=encode.astype('int')
    return encode

def horner_decode(encode,base, dim):
    arr=copy.copy(np.array(encode))
    decode=np.zeros((len(arr),dim))
    for ii in range(dim-1,-1,-1):
        digits=arr//(base**ii)
        decode[:,ii]=digits
        arr-=digits*base**ii
    return decode


def get_encode_pd(ftr_pd,ftr, base):
    mat=ftr_pd[ftr].to_numpy()
    print(mat.shape, 'base:',base)
    mat_encode=horner_encode(mat,base)  
    ftr_pd.loc[:,('encode')]=mat_encode    
    mat_decode=horner_decode(mat_encode,base,len(ftr))    
    assert np.sum(mat_decode-mat)==0    
    return mat_encode


def get_train_pd(star_dict):
    pds=[]
    train_pd=None
    for key,val in star_dict.items():
        val['label']=key
        pds.append(val)
    train_pd = pd.concat(pds,axis=0)
    return train_pd

def get_umap_pd(ftr_pd, ftr):
    umapH = umap.UMAP(n_components=2)
    umap_result = umapH.fit_transform(ftr_pd[ftr].values)
    ftr_pd['u1'] = umap_result[:,0]
    ftr_pd['u2'] = umap_result[:,1]
    return None

def get_tsne_pd(ftr_pd, ftr):
    X_embedded = TSNE(n_components=2).fit_transform(ftr_pd[ftr].values)
    ftr_pd['t1'] = X_embedded[:,0]
    ftr_pd['t2'] = X_embedded[:,1]
    return None

def get_isochrone(df,dmin=-0.016,dmax=0.3):
    df1=df[['mag_u','mag_g','mag_r','mag_i','mag_z']]
    df_data=(df1-15)/(22-15)
    df_new=pd.DataFrame()
    df_new['color_ug']=(df_data['mag_u']-df_data['mag_g']).values
    df_new['color_gr']=(df_data['mag_g']-df_data['mag_r']).values
    df_new['color_ri']=(df_data['mag_r']-df_data['mag_i']).values
    df_new['color_iz']=(df_data['mag_i']-df_data['mag_z']).values
    df_new=df_new.clip(dmin,dmax)
    df_new=(df_new-dmin)/(dmax-dmin)
    return df_data,df_new


def get_HH_pd(stream,base,ftr_len):
    count=Counter(stream)
    HH=sorted(count,key=count.get,reverse=True)
    mat_decode_HH=horner_decode(HH,base,ftr_len)
    mat_sorted=[count[HH[ii]] for ii in range(len(HH))]
    HH_pd=pd.DataFrame(np.vstack((mat_decode_HH.T,HH,mat_sorted)).T, columns=list(range(ftr_len))+['HH','freq']) 
    HH_dict={HH_pd['HH'].values[ii].astype('int'):ii for ii in range(len(HH_pd['HH'])) }
    return HH_pd,HH_dict

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
    return None

def get_unique_label(label):
    if len(label)<=2:
        return label[0]
    else:
        return 'mixed'
def get_HH_label_old(HH_pd,label_pd,encode_str='encode',label_str='label',HH_str='HH'):
    labels=[]
    for ii, HH in enumerate(HH_pd[HH_str]):
        label=list(label_pd[label_pd[encode_str]==int(HH)][label_str].unique())
        labels.append(label)
    HH_pd[label_str]=labels
    HH_pd[f'{label_str}U']=HH_pd[label_str].apply(get_unique_label)
    return None

def get_unique_label(label):
    if len(label)<=2:
        return label[0]
    else:
        return 'mixed'

def process_umap(exact_pdh, pca_comp, scale = 500):
    umapH = umap.UMAP(n_components=2)
    umap_result = umapH.fit_transform(exact_pdh[list(range(pca_comp))])
    freqlist  = exact_pdh['freq']
    lw = (freqlist/freqlist[0])**2
    u1 = umap_result[:,0]
    exact_pdh['u1'] = u1
    u2 = umap_result[:,1]
    exact_pdh['u2'] = u2
    plt.scatter(u1, u2, s = scale*lw)
    return None