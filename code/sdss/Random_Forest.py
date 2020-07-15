import os
import getpass
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import copy
import umap
from collections import Counter
import seaborn as sns
from pylab import rcParams
from sklearn.manifold import TSNE
import warnings
import time
import pandas as pd
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from prepro_umap import *


def get_RF_pd(df, intlbl=None):
    keys=[ 'u', 'g', 'r', 'i', 'z']
    ckeys=['ug', 'gr', 'ri', 'iz', 'ur', 'gi', 'rz', 'ui','gz', 'uz']
    df_color_norm=get_norm_pd(df[ckeys], r1=0.01,r2=0.99, method='ratio')
    df_norm=pd.concat([df[keys],df_color_norm],axis=1) 
    label_keys=['class','subclass','label','l8','l5']
    if intlbl is not None:
        df_label,label_dict,label_sorted=get_label_pd(df,intlbl,label_keys,intlbl=True)
    df_label,label_dict,label_sorted=get_label_pd(df,'l8',label_keys,intlbl=True)
    return df_norm, df_label,label_dict,label_sorted

def get_imp_ftr(df_norm, rfkeys, df_label, lbl,n_est=100,m_dpth=500):
    df_data=df_norm[rfkeys]
    RFlabel= pd.get_dummies(df_label[lbl])    
    t0 = time.time()
    clf = RandomForestClassifier(n_estimators = n_est,max_depth=m_dpth, random_state = 112)
    print('fitting', end='')
    clf.fit(df_data, RFlabel);
    t1 = time.time()
    t=t1-t0
    print('time', t, end='')
    ftr_imp = pd.Series(clf.feature_importances_,index=rfkeys).sort_values(ascending=False)
    sns.barplot(x=ftr_imp, y=ftr_imp.index)
    # Add labels to your graph
    plt.xlabel('Feature Importance Score')
    plt.ylabel('Features')
    plt.title("Visualizing Important Features")
    plt.legend()
    plt.show()
    return clf, ftr_imp, t

# lu_dict={}
# lbl_dict={}
# for ii, subcls in enumerate(labeldict_pd['subclass'].values):
#     lbl_dict[subcls]=labeldict_pd['T'][ii]
#     lu_dict[subcls]=labeldict_pd['L'][ii] 