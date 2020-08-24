import pandas as pd
import umap
import numpy as np
import copy
import matplotlib.pyplot as plt
from pylab import rcParams
rcParams['figure.figsize'] = 16, 10
from PIL import Image,ImageChops,ImageDraw
from sklearn.cluster import KMeans,SpectralClustering
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import ListedColormap,from_levels_and_colors
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.patches import Patch
from matplotlib.pyplot import imshow
from multiprocessing import Pool
import seaborn as sns
N_cluster=8
cmap_k=sns.color_palette('muted', N_cluster)

# IMG_PATH='/home/swei20/cancerHH/AceCanZ/cancer/lbl_data/M21_1_[46163,12853]_label.png'

IMG_PATH='/home/swei20/cancerHH/AceCanZ/data/cancer_lbl/M21_1_[46163,13653]_label.png'
lbl = np.array(Image.open(IMG_PATH))
lbl[lbl==192]=128
l128=1*((lbl==128)| (lbl==192))
l64=1*(lbl==64)

def get_kmean_cluster_plot(HH_pdh,pred_k,N_cluster,cmap_k):
    f, (ax0,ax1)=plt.subplots(1,2,figsize=((16,5)))
    sns.scatterplot('u1','u2',data=HH_pdh,hue=f'k{N_cluster}', marker='x',s=5, palette=cmap_k,legend="full",ax = ax0)
    cmap, norm =from_levels_and_colors(list(range(N_cluster+2)),[[0.99,0.99,0.99,1]]+cmap_k)
    im=ax1.imshow(pred_k, cmap=cmap, norm=norm)
    divider = make_axes_locatable(ax1)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    f.colorbar(im, cax=cax)


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

def get_intensity_hist(intensity,N_bins = 100 ,N_sigma = 3,ax=None):
    if ax is None: ax=plt.gca()
    para = np.log(intensity[intensity > 0])
    histdata = plt.hist(para, bins = N_bins, color = 'gold')
    x, y = histdata[0], histdata[1][:-1]
    y = (y[1]-y[0])/2 + y
    assert len(x) == len(y)
    x_max =  np.max(x)
    x_half = x_max//2
    mu = y[x == x_max]
    sigma = abs(y[abs(x - x_half).argmin()] -mu)
    cutoff_log = N_sigma* sigma + mu
    cutoff = int(np.exp(cutoff_log))
    peak_val = int(np.exp(mu))
    lower_sig = int(np.exp(mu - sigma))
    upper_sig = int(np.exp(mu + sigma))
    ax.axvline(mu, color = 'b', label = f'Peak = {peak_val}')
    ax.axvline(mu - sigma, color = 'cyan', ls = ':', label = f'-sigma = {lower_sig}')
    ax.axvline(mu+sigma, color = 'cyan', ls = ':', label = f'+sigma = {upper_sig}')
    ax.axvline(cutoff_log, color = 'r', label = f'{N_sigma}sigma = exp{np.round(cutoff_log,1)} =  {cutoff}' )
    ax.axhline(x_half, color = 'cyan', ls = ':')
    ax.set_title('Histogram of Intensity Cutoff ')
    ax.set_xlabel('log(intensity)')
    ax.legend()
    return cutoff

def plot_normal(ax=None,lbl=l64,c='deepskyblue',a=1):
    if ax is None: ax=plt.gca()
    ax.contour(lbl, colors=c, alpha=a, linewidths=0.5)    
    return None
def plot_cancer(ax=None,lbl=l128,c='crimson',a=1):
    if ax is None: ax=plt.gca()
    ax.contour(lbl, colors=c, alpha=a,linewidths=0.5)    
    return None

def plot_CN_all(pred,idxs, l128,l64, cmaps):
    l=len(idxs)
    f, axs=plt.subplots(l,2,figsize=(20,8*l))
    for ii, (ax0,ax1) in enumerate(axs):
        mat= (pred==idxs[ii])
        ax0.matshow(mat,cmap=ListedColormap([[0.95,0.95,0.95,1], cmaps[ii]]))
        plot_normal(ax=ax0,lbl=l64,c='deepskyblue',a=1)
        ax1.matshow(mat,cmap=ListedColormap([[0.95,0.95,0.95,1], cmaps[ii]]))
        plot_cancer(ax=ax1,lbl=l128,c='crimson',a=1)
    return None
    
def plot_2_clusters_1(final_c10, idx1 =1, idx2 = 7):
    s1 = (final_c10 == idx1)*(1)
    s2 = (final_c10 == idx2)* 1
    masked = s1 - s2
    plt.matshow(np.reshape(masked, (1004,1344)))

def get_rank(HH_pd):
    HH_pd['rk']=HH_pd['freq'].cumsum()    
    plt.figure(figsize=(8,6))
    plt.plot(HH_pd['rk']/HH_pd['rk'].values[-1])
    plt.xlabel('rank')
    plt.ylabel('ratio')
    plt.ylim(0,1)
    plt.grid()
    
def get_umap_pd(ftr_pd, ftr, n_comp=2):
    umapT = umap.UMAP(n_components=n_comp,min_dist=0.0,n_neighbors=50,metric='euclidean',random_state=1178)
    try: df_umap=ftr_pd[ftr]
    except: df_umap=ftr_pd[list(map(str,ftr))]
    umap_result = umapT.fit_transform(df_umap.values)
    for ii in range(n_comp):
        ftr_pd[f'u{ii+1}'] = umap_result[:,ii]
    sns.scatterplot('u1','u2', data=ftr_pd, marker='x',s=10, color='k')
    return umapT

def plot_clusters(stream_1D, mask, pd,  k_cluster, val = 'val', bg = -1, color = 0, sgn = 1 ):
    mask=mask.astype('bool')
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

# umapT=get_umap_pd(HH_pdc,list(range(pca_comp)))
# plt.figure(figsize=(20,5))
# sns.scatterplot('u1','u2',data=HH_pdc, hue=np.log(HH_pdc['freq']), size=np.log(HH_pdc['freq']))
    
# g = sns.PairGrid(norm_pd1k,hue='h3')
# g.map_diag(sns.kdeplot)
# g.map_offdiag(sns.kdeplot, n_levels=4);
# pcag = sns.PairGrid(HH_pdc,hue='m4',palette="Set2",
#                  hue_kws={"marker": ["o", "s", "D"]})
# pcag = sns.PairGrid(HH_plot,hue='m4',palette="Set2")
# pcag.map_diag(plt.hist,histtype="step", linewidth=3)
# pcag.map_offdiag(sns.kdeplot, n_levels=3);
# g = sns.PairGrid(color_pct,hue="lbl")
# g = g.map_upper(sns.scatterplot)
# g = g.map_lower(sns.kdeplot,n_levels=3)
# g = g.map_diag(sns.kdeplot, lw=2 )
# g.add_legend()


# f, (ax0,ax1,ax2)=plt.subplots(1,3,figsize=(20,8))
# ax0.scatter(HH_pdc['u1'],HH_pdc['u2'],s=lw**2*100, c='k')
# ax1.scatter(HH_pdc['u1'],HH_pdc['u2'],s=lw**2*100, c=HH_pdc['k8'])
# # sns.scatterplot('u1','u2',data=HH_pdc, hue='k3',ax=ax2,palette="viridis")
# ax2.scatter(HH_pdc['u1'],HH_pdc['u2'],s=lw**2*100, c=HH_pdc['m4'])

# f,(ax0,ax1)=plt.subplots(1,2,figsize=(16,5))
# ax0.scatter(HH_pdc['u1'],HH_pdc['u2'],s=lw**2*100, c=HH_pdc['m4'])
# ax1 = sns.kdeplot(HH_pdc[HH_pdc['m4']==2]['u1'],HH_pdc[HH_pdc['m4']==2]['u2'],
#                  cmap="Reds", shade=True, shade_lowest=False,label='TN')
# ax1 = sns.kdeplot(HH_pdc[HH_pdc['m4']==4]['u1'],HH_pdc[HH_pdc['m4']==4]['u2'],
#                  cmap="Blues", shade=True, shade_lowest=False,label='NTN')
# ax1 = sns.kdeplot(HH_pdc[HH_pdc['m4']==3]['u1'],HH_pdc[HH_pdc['m4']==3]['u2'],
#                  cmap="Greens", shade=True, shade_lowest=False,label='MPH')
# ax1 = sns.kdeplot(HH_pdc[HH_pdc['m4']==1]['u1'],HH_pdc[HH_pdc['m4']==1]['u2'],
#                  cmap="Purples", shade=True, shade_lowest=False,label='Interface')
# ax1.legend()

def get_kmean_cluster_plot(HH_pdh,pred_k,N_cluster,cmap_k):
    f, (ax0,ax1)=plt.subplots(1,2,figsize=((16,5)))
    sns.scatterplot('u1','u2',data=HH_pdh,hue=f'k{N_cluster}', marker='x',s=5, palette=cmap_k,legend="full",ax = ax0)
    cmap, norm =from_levels_and_colors(list(range(N_cluster+2)),[[0.95,0.95,0.95,1]]+cmap_k)
    im=ax1.imshow(pred_k, cmap=cmap, norm=norm)
    divider = make_axes_locatable(ax1)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    f.colorbar(im, cax=cax)
