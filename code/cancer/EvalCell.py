import sys
import os
import math
import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
import umap

from sklearn import preprocessing
#################### K-Means ###########################
from sklearn.cluster import KMeans
import seaborn as sns
import matplotlib.pyplot as plt

def process_umap(exact_pdh, pca_comp, scale = 500, freqInt=False):
    umapH = umap.UMAP(
    n_neighbors=30,
    min_dist=0.0,
    n_components=2,
    random_state=42)
    
    umap_result = umapH.fit_transform(exact_pdh[list(range(pca_comp))])
    freqlist  = exact_pdh['freq']
    if freqInt:
        lw=freqlist
    else:
        lw = (freqlist/freqlist[0])**2
    u1 = umap_result[:,0]
    exact_pdh['u1'] = u1
    u2 = umap_result[:,1]
    exact_pdh['u2'] = u2
    plt.scatter(u1, u2, s = scale*lw)
    return None

    
def process_kmean1(exact_pdh, u1 = 'u1', u2 = 'u2', k_cluster = 'kmean', weighted = False, N_cluster = 8):
    kmeans = KMeans(n_clusters=N_cluster,algorithem=elkan)
    if weighted:
        weight = exact_pdh['freq'].values
    else:
        weight = None
    umap_result = exact_pdh.loc[:,[u1, u2 ]].values
    kmeans.fit(umap_result, sample_weight = weight)
    y_km = kmeans.predict(umap_result)
    exact_pdh[k_cluster] = y_km + 1
    # print(exact_pdh.loc[[0,1]])
    plt.figure(figsize=(10,6))
    sns.scatterplot(
        x=u1, y=u2,
        hue= k_cluster,
        palette=sns.color_palette("muted", N_cluster),
        data=exact_pdh,
        legend="full",
        # alpha=0.3
    )
    return None  

def process_kmean(exact_pdh,  N_clusters = [3,10], u1 = 'u1', u2 = 'u2', k_cluster = 'kmean', weighted = False):
    if weighted:
        weight = exact_pdh['freq'].values
    else:
        weight = None
    k_names = []
    umap_result = exact_pdh.loc[:,[u1, u2 ]].values
    l_cluster = len(N_clusters)
    print(l_cluster)
    f, axes = plt.subplots(1,l_cluster,figsize= (16,5) )
    for i in range(l_cluster):
        N_cluster = N_clusters[i]
        k_name = f'k{N_cluster}'
        k_names += [k_name]
        kmeans = KMeans(n_clusters=N_cluster)
        kmeans.fit(umap_result, sample_weight = weight)
        y_km = kmeans.predict(umap_result)
        y_km += 1
        exact_pdh[k_name] = y_km 
        sns.scatterplot(
            x=u1, y=u2,
            hue= k_name ,
            palette=sns.color_palette("muted", N_cluster),
            data=exact_pdh,
            legend="full",
            ax = axes[i]
            # alpha=0.3 
            )
    # print(exact_pdh.loc[[0,1]])
    return k_names

def plot_clusters(stream_1D, mask, pd,  k_cluster, val = 'val', bg = -1, color = 0, sgn = -1 ):
    HHvals = np.array(pd[val])
    HHcluster = np.array(pd[k_cluster])
    masked_streams = np.zeros(np.shape(stream_1D))
    for idx, val in enumerate( HHvals): 
        label = HHcluster[idx]
        masked_streams = np.where(stream_1D != val, masked_streams, color -sgn * label)
    final_umap = np.ones(mask.shape) * bg
    final_umap[mask] = masked_streams
    return final_umap, masked_streams 


def plot_kmean_binary(stream_1D, mask, HHvals, val = 'val',  bg = -1, color = 1 ):
    masked_streams = np.zeros(np.shape(stream_1D))
    for val in HHvals: 
        masked_streams = np.where(stream_1D != val, masked_streams, color)
    final_umap = np.ones(mask.shape) * bg
    # binary_umap = np.ones(mask.shape) * bg
    # masked_binary = (masked_streams > 0) * 1
    final_umap[mask] = masked_streams
    # binary_umap[mask] = masked_binary
    return final_umap, masked_streams 


############################## Final Filtered Results #####################################
def plot_HH_masked(stream_1D, mask, cluster_dict, bg, c_ini = 5, c_fin = 20):
    masked_streams = np.zeros(np.shape(stream_1D))    
    print('ini,fin', c_ini, c_fin )
    for i in range(c_ini, c_fin ):
        masked_stream_1D = np.zeros(np.shape(stream_1D)) 
        vals = np.array(cluster_dict[f'{i + 1}'])
        for val in vals:
            masked_stream_1D = np.where(stream_1D != val, masked_stream_1D, c_fin - i)
        masked_streams += masked_stream_1D
    print(np.min(masked_streams),np.max(masked_streams))
    final_results = np.ones(mask.shape) * bg
    binary_results = np.ones(mask.shape) * bg
    masked_binary = (masked_streams > 0) * 1
    print(np.max(masked_binary))
    final_results[mask] = masked_streams
    binary_results[mask] = masked_binary
    return final_results, binary_results

def plot_2_clusters_1(final_c10, idx1 =1, idx2 = 7):
    s1 = (final_c10 == idx1)*(1)
    s2 = (final_c10 == idx2)* 1
    masked = s1 - s2
    plt.matshow(np.reshape(masked, (1004,1344)))
    # plt.title(f'cluster yellow: {idx1}  and blue: {idx2}')
#     plt.savefig(f'{name}/p{idx1}m{idx2}.png', dpi = 1000)

def plot_4_clusters(final_c10, idx1, idx2, idx3, idx4):
    s1 = (final_c10 == idx1)*(1)
    s2 = (final_c10 == idx2)* 1
    s3 = (final_c10 == idx3)*(2)
    s4 = (final_c10 == idx4)*(2)
    masked = s1 + s3 - s2 - s4
    plt.matshow(np.reshape(masked, (1004,1344)))
    plt.title(f'cluster Green: {idx1} , Yellow: {idx3}, Blue: {idx2}, Purple:{idx4}')
    # plt.savefig(f'{name}/p{idx1}pp{idx3}m{idx2}mm{idx4}.png', dpi = 1000)

    return None