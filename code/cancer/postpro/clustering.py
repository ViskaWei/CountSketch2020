import pandas as pd
import umap
import numpy as np
import copy
import seaborn as sns
# from sklearn.manifold import TSNE
from sklearn.cluster import KMeans,SpectralClustering
import matplotlib.pyplot as plt
from pylab import rcParams
rcParams['figure.figsize'] = 16, 10

# clustering = SpectralClustering(n_clusters=5, assign_labels="discretize", random_state=0).fit(X)
# colors = np.array(['#377eb8', '#ff7f00', '#4daf4a','#f781bf', '#a65628', '#984ea3', '#999999', '#e41a1c', '#dede00'])
# print(colors[clustering.labels_])

def process_kmean(exact_pdh,  N_clusters = [3,10], u1 = 'u1', u2 = 'u2', k_cluster = 'kmean', weight=None):
    k_names = []
    umap_result = exact_pdh.loc[:,[u1, u2 ]].values
    l_cluster = len(N_clusters)
    f, axes = plt.subplots(1,l_cluster,figsize= (16,5) )
    for i in range(l_cluster):
        N_cluster = N_clusters[i]
        k_name = f'k{N_cluster}'
        k_names += [k_name]
        kmeans = KMeans(n_clusters=N_cluster,n_init=30, algorithm='elkan')
        kmeans.fit(umap_result, sample_weight = weight)
        exact_pdh[k_name] = kmeans.labels_ +1
        sns.scatterplot(
            x=u1, y=u2,
            hue= k_name , marker='x',s=5,
            palette=sns.color_palette("muted", N_cluster),
            data=exact_pdh,
            legend="full",
            ax = axes[i]
            # alpha=0.3 
            )
    # print(exact_pdh.loc[[0,1]])
    return k_names,kmeans

def plot_clusters(stream_1D, mask, pd,  k_cluster, val = 'val', bg = -1, color = 0, sgn = 1 ):
    HHvals = np.array(pd[val])
    HHcluster = np.array(pd[k_cluster])
    masked_streams = np.zeros(np.shape(stream_1D))
    for idx, val in enumerate( HHvals): 
        label = HHcluster[idx]
        masked_streams = np.where(stream_1D != val, masked_streams, color +sgn * label)
    final_umap = np.ones(mask.shape) * bg
    final_umap[mask] = masked_streams
    return final_umap, masked_streams 

def get_conf_mat(pred, target):
    pred=pred.reshape((1004,1344))
    conf_mat = pd.DataFrame(index=np.unique(pred), columns=np.unique(target))
    conf_mat[:] = 0
    for i in np.unique(pred):
        for j in np.unique(target):
            conf_mat.loc[i,j] = np.sum((pred == i)&(target==j))
    print(conf_mat)
#     conf_mat['ratio']=conf_mat.apply(lambda x: x[[64]]/x[[128]])
    return conf_mat

def get_freq_aug(df):
    freq=df['freq'].values
    df['FN']=freq/freq[-1]
    df['FR']=df['FN'].apply(lambda x: 1+np.floor(np.log2(x)))
#     df['RR']=1+np.floor(np.log2(freq/freq[-1]))
    return freq
def get_aug_pd(df,ftr,mode='FR'):
    data=df[ftr].values   
    freqN =df[mode].values
    freqInt=freqN.astype('int')
    plt.figure(figsize=(5,5))
    _=plt.hist(freqInt,bins=freqInt[0])  
    data_aug=data[0]
    freq_list=[]
    np.random.seed(112)
    for ii, da in enumerate(data[1:]):
        freqInt_ii=freqInt[ii]
        freq_list+=[freqInt_ii]*freqInt_ii
        randmat=np.random.rand(freqInt_ii,pca_comp)-0.5
        da_aug = da+0.25*randmat
        assert np.sum(np.round(da_aug)-da)<0.001
        data_aug=np.vstack((data_aug,da_aug))
    data_aug=data_aug[1:]
    print(data_aug.shape,len(freq_list))
    aug_pd=pd.DataFrame(data_aug, columns=list(range(pca_comp)))
    aug_pd['freqInt']=freq_list
    aug_pd['freq']=aug_pd['freqInt'].apply(lambda x: float(x))    
    return aug_pd