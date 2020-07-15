import pandas as pd
import umap
import numpy as np
import copy
# from sklearn.manifold import TSNE
from collections import Counter
import joblib

model_dict={}
def get_mapping_pd(ftr_pd,umapT,ftr):
    try: df_umap=ftr_pd[ftr]
    except: df_umap=ftr_pd[list(map(str,ftr))]
    df_umap1=(df_umap.sample(1)).values
    assert (np.round(df_umap1)==df_umap1).all()
    u_da=umapT.transform(df_umap.values)
    # umapped=pd.DataFrame(u_da, columns=[f"u{ii}" for ii in range(len(u_da[0]))])
    # ftr_pd=pd.concat([ftr_pd,umapped],axis=1)
    for ii in range(len(u_da[0])):
        ftr_pd[f'u{ii+1}'] = u_da[:,ii]
    print(ftr_pd.keys())
    return ftr_pd
    
def get_HH_subclass_label(HH_pd,label_pd,encode_str='encode',label_str='label',HH_str='HH'):
    labels=[]
    for ii, HH in enumerate(HH_pd[HH_str]):
        label_list=label_pd[label_pd[encode_str]==int(HH)][label_str]
        c=Counter(label_list)
        labels.append(c.most_common(1)[0][0])
    HH_pd[f'{label_str}m']=labels
    return None
    
###############################SPEC_UMAP###############################

def get_umap_pd(ftr_pd, ftr, n_comp):
    umapT = umap.UMAP(n_components=n_comp,min_dist=0.0,n_neighbors=50, random_state=1178)
    try: df_umap=ftr_pd[ftr]
    except: df_umap=ftr_pd[list(map(str,ftr))]
    umap_result = umapT.fit_transform(df_umap.values)
    for ii in range(n_comp):
        ftr_pd[f'u{ii+1}'] = umap_result[:,ii]
    return umapT

def get_spec_mapping(HH_pd,ftr, df_lbl, base,name,umap_comp,HH_cut=20000):
    df_label=copy.deepcopy(df_lbl)
    HH_pdc=HH_pd[:HH_cut]
    print(len(HH_pdc),len(HH_pd),HH_pdc['freq'].sum(), HH_pdc['freq'][0],HH_pdc['freq'].tail())
    umapT=get_umap_pd(HH_pdc, list(range(len(ftr))), umap_comp)
    for ii in ['class','subclass','l5','l8']:
        get_HH_subclass_label(HH_pdc,df_label,encode_str='encode',label_str=ii,HH_str='HH') 
    return HH_pdc,umapT

def get_spec_label(HH_pd, df_lbl):
    df_label=copy.deepcopy(df_lbl)
    for ii in ['class','subclass','l5','l8']:
        get_HH_subclass_label(HH_pd,df_label,encode_str='encode',label_str=ii,HH_str='HH') 
    return HH_pd


# def get_spec_mapping(HH_pd,ftr, df_lbl, base,name,HH_cut=20000):
#     df_label=copy.deepcopy(df_lbl)
#     # lb,ub=int(HH_pd['freq'][0]*lbr),int(HH_pd['freq'][0])
#     # HH_pdc=HH_pd[HH_pd['freq']>lb]
#     HH_pdc=HH_pd[:HH_cut]
#     print(len(HH_pdc),len(HH_pd),HH_pdc['freq'].sum(), HH_pdc['freq'][0],HH_pdc['freq'].tail())
#     get_HH_label(HH_pdc,df_label,encode_str='encode',label_str='class',HH_str='HH') 
#     HH_pdQS,QS_stats=get_QS_stats(HH_pdc,model_dict,name,pd_key='classU',clusters_str=['QSO','STAR','mixed']) 
#     umapT=get_umap_pd(HH_pdQS, list(range(len(ftr))));
# #     sns.scatterplot('u1','u2',data=HH_pdQS)
#     get_HH_label(HH_pdQS,df_label,encode_str='encode',label_str='l5',HH_str='HH')    
#     HH_pd5=HH_pdQS[HH_pdQS[f'l5U']!='mixed']
#     stat25=(len(HH_pd5),len(HH_pdQS), len(HH_pd5)/len(HH_pdQS))
#     model_dict[name]['s25']=stat25
#     print(stat25)
#     get_HH_label(HH_pdQS,df_label,encode_str='encode',label_str='l8',HH_str='HH')    
#     HH_pd8=HH_pdQS[HH_pdQS[f'l8U']!='mixed']
#     stat58=[len(HH_pd8),len(HH_pdQS), len(HH_pd8)/len(HH_pdQS)]
#     model_dict[name]['s58']=stat58
#     print(stat58)
#     model_dict[name]['HH']=[len(HH_pdc),len(HH_pd),HH_pdc['freq'].sum(), HH_cut]
#     return HH_pdQS,umapT, model_dict

# def get_spec_label(HH_pd, df_lbl,name,HH_cut=20000):
#     df_label=copy.deepcopy(df_lbl)
#     # lb,ub=int(HH_pd['freq'][0]*lbr),int(HH_pd['freq'][0])
#     # HH_pdc=HH_pd[HH_pd['freq']>lb]
#     HH_pdc=HH_pd[:HH_cut]
#     print(len(HH_pdc),len(HH_pd),HH_pdc['freq'].sum(), HH_pdc['freq'][0],HH_pdc['freq'].tail())
#     get_HH_label(HH_pdc,df_label,encode_str='encode',label_str='class',HH_str='HH') 
#     # HH_pdQS,QS_stats=get_QS_stats(HH_pdc,model_dict,name,pd_key='classU',clusters_str=['QSO','STAR','mixed']) 
#     model_dict={name:{}}
#     HH_pdQS=HH_pdc
#     get_HH_label(HH_pdQS,df_label,encode_str='encode',label_str='l5',HH_str='HH')    
#     HH_pd5=HH_pdQS[HH_pdQS[f'l5U']!='mixed']
#     stat25=(len(HH_pd5),len(HH_pdQS), len(HH_pd5)/len(HH_pdQS))
#     model_dict[name]['s25']=stat25
#     print(stat25)
#     get_HH_label(HH_pdQS,df_label,encode_str='encode',label_str='l8',HH_str='HH')    
#     HH_pd8=HH_pdQS[HH_pdQS[f'l8U']!='mixed']
#     stat58=[len(HH_pd8),len(HH_pdQS), len(HH_pd8)/len(HH_pdQS)]
#     model_dict[name]['s58']=stat58
#     print(stat58)
#     model_dict[name]['HH']=[len(HH_pdc),len(HH_pd),HH_pdc['freq'].sum(), HH_cut]
#     return HH_pdQS, model_dict

# def get_QS_stats(HH_pd,model_dict,name,pd_key='classU',clusters_str=['QSO','STAR','mixed']):
#     HH_pdh=HH_pd[(HH_pd['class']=='STAR')|(HH_pd['class']=='QSO')|(HH_pd['class']=='QSOSTAR')]
#     N_HH=HH_pdh['freq'].sum()
#     HH_pdQS=HH_pdh[(HH_pdh['classU']!='mixed')]
#     N_Q=(HH_pdh[pd_key]==clusters_str[0]).sum()
#     N_S=(HH_pdh[pd_key]==clusters_str[1]).sum()
#     N_M=(HH_pdh[pd_key]==clusters_str[2]).sum()
#     N_QS, N_A=len(HH_pdQS), len(HH_pdh)
#     print('Heaviness:', N_QS, N_A, N_HH)
#     assert (abs(N_QS-N_Q-N_S)<1e-5) and( abs(N_M+N_QS-N_A)<1e-5)
#     e, p_Q, p_S= N_QS/N_A, N_Q/(N_Q+N_M), N_S/(N_S+N_M)
#     print(f'model:{name} e:{np.round(e*100,1)}%, Q purity: {np.round(p_Q*100,1)}%, S purity: {np.round(p_S*100,1)}%')
#     QS_stats=[e,p_Q,p_S,N_HH]
#     model_dict[name]={"QS":QS_stats}
#     return HH_pdQS,QS_stats

# def get_HH_label(HH_pd,label_pd,encode_str='encode',label_str='label',HH_str='HH'):
#     labels=[]
#     ulabels=[]
#     for ii, HH in enumerate(HH_pd[HH_str]):
#         label=label_pd[label_pd[encode_str]==int(HH)][label_str].unique()
#         lblstr="".join(label.astype(str))
#         labels.append(lblstr)
#         if len(label)>1:
#             ulabels.append('mixed')
#         else:
#             ulabels.append(lblstr)
#     HH_pd[label_str]=labels
#     HH_pd[f'{label_str}U']=ulabels
#     return HH_pd

    # f,axs=plt.subplots(1,3,figsize=(20,8))
    # hues=['classU','l5U','l8U']
    # for ii, ax in enumerate(axs.flatten()):
    #     sns.scatterplot('u1','u2',data=HH_pdQS,hue=hues[ii],ax=ax)
    #     if ii==0: ax.set_title(f'{name}, base{base}', fontsize=16)
    #     if ii==1: ax.set_title(f'{ftr}', fontsize=16)