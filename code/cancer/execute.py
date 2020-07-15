# -*- coding: utf-8 -*-
import sys
import os
import math
import numpy as np
import pandas as pd
import json
import copy
import matplotlib.pyplot as plt
import umap.umap_ as uma
from collections import Counter
# sys.path.insert(0, r'C:\Users\viska\Documents\AceCan')
# os.chdir(r"C:\Users\viska\Documents\AceCan")
# data_dir = r'.\bki'
data_dir = './bki'
output_path= os.getcwd()
from timeit import default_timer as timer
from data.dataset import Dataset
from concurrent.futures import ThreadPoolExecutor, as_completed
from sklearn import preprocessing


############## Process Data #################
def process_dataset_pc(data_dir, base, num, pca_comp, topk, test,smth=False):   
    images = Dataset(data_dir, num)
    num  = images.N_img  #in case: num > N_images
#     pca_combined = np.zeros([images.layer, images.layer])
    print(f'Processing {num} [test :{test}] images with original size {images.size} ')
    with ThreadPoolExecutor() as executor: 
        futures = []
        for idx in range(num):
            futures.append(executor.submit(lambda x: run_step_multiple(x, images, smth, test), idx))
            # print(f" No.{idx} image is loaded")
        mul_comb = np.zeros([images.layer,images.layer])
        for future in as_completed(futures):
            mul = future.result()
            mul_comb += mul
        pc = run_step_pc(mul_comb, pca_comp)
    return images.data1D, pc

def run_step_multiple(idx, images, smth, test = 'False'):
    if test:
        images.get_test_data(idx, smth)
    else:
        images.get_data(idx, smth)
    images.data1D[idx] = np.reshape(images.data[idx], [images.ver*images.hor, images.layer]).astype(float)
    return images.data1D[idx].T.dot(images.data1D[idx])

def run_step_pc(mul_comb, pca_comp):
    # use svd since its commputational faster
    print("=============== run step PCA ===============")
    u,s,v = np.linalg.svd(mul_comb)
    assert np.allclose(u, v.T)
    print('Explained Variance Ratio', np.round(s/sum(s),3))
    pc = u[:,:pca_comp]
    return pc

def process_pca(data1Ds, pc, num, pca_comp, N_bins = 100, N_sigma = 3 ):
    pca_results = run_step_pc_transform(0, data1Ds, pc)
    # results = data1Ds[0]
    for i in range(1,num):
        pca_result = run_step_pc_transform(i, data1Ds, pc)
        pca_results = np.vstack((pca_results,pca_result))
        # results = np.vstack((results, data1Ds[i]))
    # pca_results = pca_results[1:,:]
    print('========= Intensity ==============')
    intensity = (np.sum(pca_results**2, axis = 1))**0.5
    # cutoffH = np.mean(intensity).round()
    return intensity, pca_results

def run_step_pc_transform(x, data1Ds, pc):
    return data1Ds[x].dot(pc)
    
#===============================intensity=====================================
def process_intensity(pca_result, intensity, base, pca_comp, cutoff = None):
    if cutoff == None:
        cutoff = run_step_cutoff(intensity)
    mask = intensity > cutoff
    print('norm length',np.sum(mask))
    norm_data = np.divide(pca_result[mask], intensity[mask][:,None])
    print('norm_data', np.min(norm_data), np.max(norm_data))
    pca_rebin = np.trunc((norm_data + 1) * base/2)
    print('rebin, min/mac', np.min(pca_rebin), np.max(pca_rebin))
    stream_1D = 0
    for comp in range(pca_comp):
        stream_1D = stream_1D + pca_rebin.T[comp]*base**comp
    return stream_1D, norm_data, mask

def run_step_cutoff(intensity, N_bins = 100, N_sigma = 3 ):
    para = np.log(intensity[intensity > 0])
    (x,y) = np.histogram(para, bins = N_bins)
    y = (y[1]-y[0])/2 + y[:-1]
    assert len(x) == len(y)
    x_max =  np.max(x)
    x_half = x_max//2
    mu = y[x == x_max]
    sigma = abs(y[abs(x - x_half).argmin()] -mu)
    cutoff_log = N_sigma* sigma + mu
    cutoff = int(np.exp(cutoff_log).round())
    return cutoff

def run_step_norm(pca_result, intensity, cutoffH, pca_comp):
    mask = intensity > cutoffH
    # print('norm length',np.sum(mask))
    uniform_data = np.divide(pca_result[mask], intensity[mask][:,None])
    df_uni=pd.DataFrame(uniform_data, columns=list(range(len(pca_comp))))
    print('df_uni',df_uni)
    df_norm=get_norm_pd(df_uni,r=0.01,std=False)
    print('df_norm',df_norm.describe())
    # print('norm_data', np.min(norm_data), np.max(norm_data))
    return df_norm

def run_step_encode(df_norm, base,dtype):
    mat=(df_norm*(base-1)).round()
    assert (mat.min().min()>=0) & (mat.max().max()<=base-1)
    mat_encode=horner_encode(mat,base,dtype) 
    mat_decode=horner_decode(mat_encode,base,len(mat.keys()),dtype)  
    try:
        assert np.sum(abs(mat_decode-mat.values))<=0.0001    
    except:
        print(np.nonzero(np.sum(abs(mat_decode-mat.values),axis=1)), np.sum(abs(mat_decode-mat.values)))
        raise 'overflow, try lower base or fewer features'     
    return mat_encode

def horner_encode(mat,base,dtype):
    r,c=mat.shape
    print('samples:',r,'ftrs:',c, 'base:',base)
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

def get_norm_pd(df,r=0.01,std=False): 
    df1=(df-df.mean())/df.std() if std else df
    vmin=df1.quantile(r)
    vrng=df1.quantile(1-r)-vmin
    df_new=((df1-vmin)/vrng).clip(0,1)
    return df_new

#################### Intensity ########################

def get_intensity_hist(intensity,N_bins = 100 ,N_sigma = 3):
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
    plt.axvline(mu, color = 'b', label = f'Peak = {peak_val}')
    plt.axvline(mu - sigma, color = 'cyan', ls = ':', label = f'-sigma = {lower_sig}')
    plt.axvline(mu+sigma, color = 'cyan', ls = ':', label = f'+sigma = {upper_sig}')
    plt.axvline(cutoff_log, color = 'r', label = f'3sigma = exp{np.round(cutoff_log,1)} =  {cutoff}' )
    plt.axhline(x_half, color = 'cyan', ls = ':')
    plt.title('Histogram of Intensity Cutoff ')
    plt.xlabel('log(intensity)')
    plt.legend()
    return cutoff


#################### 1D Stream #####################
def inverse_mapcode(stream_1D, base, pca_comp):
    stream_1D = np.array(stream_1D)
    inverted_pca = np.zeros((pca_comp,len(stream_1D)), dtype='uint64')
    for i in range(pca_comp):
        inverted_pca[i] = stream_1D % base
        stream_1D = stream_1D // base
    return inverted_pca

def process_stream_1D(stream_1D, base, pca_comp, topk ):
    c = Counter(stream_1D)
    HH = c.most_common(topk)
    exact_a = np.array(HH)
    exact_val, exact_freq = exact_a[:,0].astype('Int64'), exact_a[:,1].astype('Int64')
    exact_pca = inverse_mapcode(exact_val, base,  pca_comp)
    exact_pd = pd.DataFrame(exact_pca.T, columns = range(pca_comp))
    exact_pd['freq'] = np.abs(exact_freq)
    exact_pd['val'] = exact_val
    if len(exact_pd) > 10000:
        print('exact_pd exceeding 10000, output top 10000 only')
        return exact_pd[:10000]
    # np.savetxt(f'{name}/exact_pdh', exact_pdh)
    return exact_pd

# def process_stream_1D0(stream_1D, base, pca_comp, topk, percentage = 0.01 ):
#     c = Counter(stream_1D)
#     HH = c.most_common(topk)
#     exact_a = np.array(HH)
#     exact_val, exact_freq = exact_a[:,0].astype('Int64'), exact_a[:,1].astype('Int64')
#     exact_pca = inverse_mapcode(exact_val, base,  pca_comp)
#     exact_pd = pd.DataFrame(exact_pca.T, columns = range(pca_comp))
#     exact_pd['freq'] = np.abs(exact_freq)
#     exact_pd['val'] = exact_val
#     high_cut = exact_pd['freq'][0]* percentage
#     print(high_cut)
#     exact_pdh = exact_pd[exact_pd['freq']> high_cut]
#     print('#exact_pdh', len(exact_pdh), high_cut)
#     print(exact_pd)
#     # np.savetxt(f'{name}/exact_pdh', exact_pdh)
#     return exact_pdh, exact_pd

#################### UMAP ###########################
def process_umap(exact_pdh, pca_comp, scale = 500):
    umapH = uma.UMAP()
    umap_result = umapH.fit_transform(exact_pdh[list(range(pca_comp))])
    freqlist  = exact_pdh['freq']
    lw = (freqlist/freqlist[0])**2
    u1 = umap_result[:,0]
    exact_pdh['u1'] = u1
    u2 = umap_result[:,1]
    exact_pdh['u2'] = u2
    plt.scatter(u1, u2, s = scale*lw)
    return None

#################### Testing #######################

def test_mul_comb(mul_comb, pc, pca_comp):
    u,s,v = np.linalg.svd(mul_comb)
    assert np.allclose(u[:,:pca_comp], pc)
    plt.plot(np.log(s))
    plt.ylabel('log(eigenvalues)')
    plt.xlabel('layers')

# def test_rebin(self, norm_data, mask, base, idx = 0, bg = -0.1):
#     masked_rebin = np.ones([pixel * num, pca_comp])* (bg)
#     pca_rebin = np.trunc((norm_data + 1) * base/2)
#     print('rebin, min/mac', np.min(pca_rebin), np.max(pca_rebin))
#     masked_rebin[mask] = pca_rebin
#     plot_rebin_data(self, masked_rebin, idx, bg, base)
#     return masked_rebin


############################## Count Sketch #####################################

def process_countsketch(d, stream_1D, base, topk, col_range, row_range, device = 'cuda'):
    sketchs = {}
    for row in row_range:     
        for col in col_range:
            val,freq = run_step_sketch(stream1D, d, col,row, topk, device)
            sketchs[f'{row}_{col}_val'] = val
            sketchs[f'{row}_{col}_freq'] = freq
    return val,freq
# val,freq = process_countsketch(d, vec, base, topk, col_range, row_range)

def run_step_sketch(stream_1D, d, c,r,k, device = 'cuda'):
    csv = CSVec(d,c,r,k, device)
    stream_1D_tr = torch.tensor(stream_1D, dtype=torch.int64)
    cs.accumulateVec_heap(stream_1D_tr)
    cs_topk = csv.topk.cpu().numpy()
    cs_topk = cs_topk[cs_topk[:,0]>0]
    return cs_topk[:,0],cs_topk[:,1]




# def process_eigen()

def main():   
    # sys.path.insert(0, r'C:\Users\viska\Documents\AceCan')
    # os.chdir(r"C:\Users\viska\Documents\AceCan")
    # Output_path = r"C:\Users\viska\Documents\AceCan\UMAP"
    try:
        os.mkdir(Output_path)
    except:
        None
    load_data = 1
    num = 40
    base = 8
    topk = 10000
    col_range =  [8000]
    row_range = [5]
    display = 20
    pca_comp = 8
    d = base ** pca_comp 
    data_dir = r'.\bki'    
    test = 0
    name = os.path.join(Output_path, f'b{base}topk{topk}N{num}test')
    if load_data == True:
        try:
            os.mkdir(name)
        except:
            print('overwriting directory')
        print(f'Output directory: {name}')
        print('################################## Running PCA Preprocessing ###################################')
#          exact_HH, vec, 
        stream_concat = process_dataset(data_dir, base, num, pca_comp, topk, test)
#         np.savetxt(f'{name}\\exact.txt', exact_HH)
#         np.savetxt(f'{name}\\vec_b{base}.txt',vec)   
#         np.savetxt(f'{name}\\stream_concat{base}.txt',stream_concat) 
#         sketch_times = process_countsketch(d, vec, base, topk, col_range, row_range, display, name)
#         print(sketch_times)
    elif load_data == False:
        # load preprocessed data
        print('################################## Loading 1D frequency vector ###################################')
#         with open(f'{name}\\vecs_b{base}.txt') as json_file:
#             vec = json.load(json_file)  
        vec = np.loadtxt(f'{name}\\vec_b{base}.txt')
        print('vec',vec)
        sketch_times = process_countsketch(d, vec, base, topk, col_range, row_range, display, name)
        print(sketch_times)
#         print('################################## Analyzing Runtime ###################################')      
#        with open(f'{name}\\exact_time_b{base}k{topk}.txt') as json_file:
#            exact_time = json.load(json_file)
#        finish(exact_time, sketch_times)
    else:
        raise ValueError('load_data boolean value not set')
#    finish(exact_time, sketch_times)
#    plots = Plotting()
#    plots.layer_plot_4()
    
    
if __name__ == "__main__":
    main()
    

# from __future__ import print_function
# import time
# import numpy as np
# import pandas as pd
# # from sklearn.datasets import fetch_mldata
# from sklearn.decomposition import PCA
# from sklearn.manifold import TSNE
# %matplotlib inline
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
# import seaborn as sns
# from matplotlib.colors import LogNorm
# import umap.umap_ as uma
# import math
# from collections import Counter

# ############################## Glueviz Functions #########################################
# def get_cluster_idx(data0, sub_range, exact_pdh, col = 'col13'):
#     cluster = {}
#     i = 1
#     for subset in sub_range:
#         layer_data = data0.subsets[subset]
#         cluster[i] = layer_data[col].astype(int)
#         i+=1
#     exact_pdh['cluster'] = np.zeros(len(exact_pdh)).astype(int)
#     for key in range(1,max(sub_range)+2):
#         exact_pdh.loc[cluster[key],'cluster'] = int(key)
#     return None

# def plot_umap_clusters(stream_1D, exact_pdhh, bg = -1 ):
#     HHvals = np.array(exact_pdhh['val'])
#     HHcluster = np.array(exact_pdhh['cluster'])
#     masked_streams = np.zeros(np.shape(stream_1D))
#     for idx, val in enumerate( HHvals): 
#         label = HHcluster[idx]
#         masked_streams = np.where(stream_1D != val, masked_streams, label)
#     final_umap = np.ones(mask.shape) * bg
#     binary_umap = np.ones(mask.shape) * bg
#     masked_binary = (masked_streams > 0) * 1
#     final_umap[mask] = masked_streams
#     binary_umap[mask] = masked_binary
#     return final_umap, binary_umap,  masked_streams 


# #################### Merger ###########################

# def check_cluster(diff, base):
#     #TODO: check if we need to limit decode
#     if diff == 0:
#         return True
#     else:
#         decode = math.log(abs(diff), base)
#         same_cluster_query = np.floor(decode) == decode   
#         return same_cluster_query

# def get_merged_HHs(vals, base, limit):
#     merge_idx_dict = {}
#     val_l = len(vals)
#     clusters = np.zeros(val_l)
#     processed = [False] * val_l
#     cluster_dict= {}
#     color_idx = int(1 )
#     merged = False
#     for idx_1 in range(val_l):
#         if not processed[idx_1]:
#             val = vals[idx_1]
#             cluster_dict[f'{color_idx}'] = [val]
#             clusters[idx_1] = color_idx
#             for idx_2 in range(idx_1+1, val_l): 
#                 val_2 = vals[idx_2]
#                 diff = abs(val - val_2)
#                 if check_cluster(diff, base):                   
#                     if not processed[idx_2]:
#                         processed[idx_2] = True
#                     else:
#                         color_idx2 = int(clusters[idx_2])
#                         if not merged:                      
#                             if color_idx2 < color_idx:
#                                 # print(color_idx,color_idx2,val, val_2)
#                                 cluster_dict[f'{color_idx2}'] += cluster_dict[f'{color_idx}'] 
#                                 cluster_dict[f'{color_idx}']  = []
#                                 clusters[clusters == color_idx] = color_idx2
#                                 color_idx0 = color_idx
#                                 color_idx = color_idx2
#                                 merged = True
#                             else:
#                                 print('error')
#                         else:
#                             if color_idx2 != color_idx:
#                                 min_idx, max_idx = min( color_idx2, color_idx), max( color_idx2, color_idx)
#                                 if min_idx not in merge_idx_dict:
#                                     merge_idx_dict[min_idx] = set() 
#                                 merge_idx_dict[min_idx].add(max_idx)    
#                                 # color_idx, color_idx0 = min(color_idx2, color_idx), max(color_idx2, color_idx)
#                                 # # print(color_idx2, color_idx, color_idx0 )
#                                 # cluster_dict[f'{color_idx}'] += cluster_dict[f'{color_idx0}'] 
#                                 # cluster_dict[f'{color_idx0}']  = []  
#                                 # clusters[clusters == color_idx0] = color_idx                    

#                 ####################### assign values
#                     cluster_dict[f'{color_idx}'] += [val_2] 
#                     clusters[idx_2] = color_idx
#             ############################## End idx_2 for loop ##############################
#             if merged:
#                 color_idx = color_idx0
#                 merged = False
#             else:
#                 color_idx += 1  
#             # color_idx += 1
#             # print('color_idx',color_idx)
#             processed[idx_1]  = True
#         if color_idx > limit:
#             print(f'Top {limit} color clusters found at {idx_1}, to continue increase limit' )
#             break
#     print(merge_idx_dict)
#     return clusters, cluster_dict, merge_idx_dict

# def get_merged_pd(exact_pdh, base, limit = 10):
#     vals = exact_pdh['val']
#     val_l = len(vals)
#     clusters = np.zeros(val_l)
#     processed = [False] * val_l
#     color_idx = 1 
#     for idx_1 in range(val_l):
#         if not processed[idx_1]:
#             val = vals[idx_1]
#             clusters[idx_1] = color_idx
#             for idx_2 in range(idx_1+1, val_l): 
#                 val_2 = vals[idx_2]
#                 diff = abs(val - val_2)
#                 if check_cluster(diff, base):
#                     clusters[idx_2] = color_idx
#                     processed[idx_2] = True
#             color_idx += 1  
#             processed[idx_1]  = True
#         if color_idx > limit:
#             print(f'Top {limit} color clusters after merger found, to continue increase limit' )
#             break
#     return exact_pdh


#################### PCA ########################
# def process_pca_DO0(data1Ds, pc, num, pca_comp):
#     with ThreadPoolExecutor() as executor: 
#         futures = []
#         for idx in range(num):
#             futures.append(executor.submit(lambda x: run_step_pc_transform(x, data1Ds, pc), idx))
#             # print(f" No.{idx} image is transformed")
#         pca_results = np.zeros([1,pca_comp])
#         for future in as_completed(futures):
#             pca_result = future.result()
#             pca_results = np.vstack((pca_results,pca_result))
#     pca_results = pca_results[1:,:]
#     print('========= Intensity ==============')
#     intensity = (np.sum(pca_results**2, axis = 1))**0.5
#     return intensity, pca_results

# def process_pca_DO(data1Ds, pc, num, base, pca_comp, N_bins = 100, N_sigma = 3 ):
#     pca_results = run_step_pc_transform(0, data1Ds, pc)
#     results = data1Ds[0]
#     for i in range(1,num):
#         pca_result = run_step_pc_transform(i, data1Ds, pc)
#         pca_results = np.vstack((pca_results,pca_result))
#         results = np.vstack((results, data1Ds[i]))
#     # pca_results = pca_results[1:,:]
#     print('========= Intensity ==============')
#     intensity = (np.sum(pca_results**2, axis = 1))**0.5
#     # cutoffH = np.mean(intensity).round()
#     cutoffH = run_step_cutoff(intensity, N_bins = 100, N_sigma = 3 )
#     print('cutoffH is set to be mean', cutoffH)
#     stream_1D, mask = run_step_norm(pca_results, intensity, cutoffH, base, pca_comp)
#     return stream_1D, mask
