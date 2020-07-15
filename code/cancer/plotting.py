import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import seaborn as sns
from matplotlib.colors import LogNorm
import pandas as pd
#from collections import Counter 
#from CancerCell import *

class Plotting():
    '''
    class of functions for visualizing the data
    '''
    def __init__(self,idx = 1,ver = 400,hor = 300, n_ver = 1, n_hor = 8, fig_ver = 16, fig_hor = 10, txt_size = 6, color = '#34BBBB'):
        self.idx = idx
        self.n_ver = n_ver
        self.n_hor = n_hor
        self.color = color
        self.fig_hor = fig_hor
        self.fig_ver = fig_ver
        self.txt_size = txt_size
        self.ver = ver
        self.hor = hor
        self.pixel = ver * hor
        self.num = 40
        self.pca_comp = 8
        self.base = 8
        
           
    def layer_plot_4(self, data,  gap = 3, title = None):
        dmin, dmax = np.min(data), np.max(data)
        f, axs = plt.subplots(self.n_ver, self.n_hor, figsize=(self.fig_ver, self.fig_hor), dpi=120, facecolor='w', edgecolor='k')
        f.suptitle(title)
        layer_idx = 0
        for ax in axs.flatten():
            ax.matshow(data[layer_idx,:,:], vmin = dmin, vmax = dmax)
            ax.title.set_text( f'layer = {layer_idx+1}')
            ax.axis('off')
            layer_idx += gap
            
    def plot_pca_data(self, pca_data,pca_comp, title= None):
        reshaped = np.reshape(pca_data, [pca_comp, self.ver, self.hor])
#         transposed = np.transpose(reshaped,(1,2,0))
        self.layer_plot_4(reshaped, title = title, gap = 1)  

    def test_norm(self, norm_data, mask, idx= 0, bg = -2):
        masked_pca = np.ones([self.pixel * self.num, self.pca_comp])* (bg)
        masked_pca[mask] = norm_data
        self.plot_pca_data(masked_pca[self.pixel*idx:self.pixel*(idx+1),:].T, pca_comp = self.pca_comp)
        return masked_pca
        
    def test_rebin(self, norm_data, mask, idx = 0, bg = -0.1, title = None):
        masked_rebin = np.ones([self.pixel * self.num, self.pca_comp])* (bg)
        pca_rebin = np.trunc((norm_data + 1) * self.base/2)
        print('rebin, min/mac', np.min(pca_rebin), np.max(pca_rebin))
        masked_rebin[mask] = pca_rebin
        self.plot_rebin_data(masked_rebin, idx, bg, title)
        return masked_rebin

    def plot_rebin_data(self, rebin_data, idx = 26, bg = -0.1, title = None):
        dmax = np.max(rebin_data)
        rebin0 = rebin_data[self.pixel*idx:self.pixel*(idx+1),:]
        reshaped = np.reshape(rebin0.T, [self.pca_comp,self.ver, self.hor])
        f, axs = plt.subplots(self.n_ver, self.n_hor, figsize=(self.fig_ver, self.fig_hor), dpi=120, facecolor='w', edgecolor='k')
        f.suptitle(title)
        layer_idx = 0
        for ax in axs.flatten():
            ax.matshow(reshaped[layer_idx,:,:] , vmin = bg, vmax = dmax )
            ax.title.set_text( f'layer = {layer_idx+1}')
            ax.axis('off')
            layer_idx += 1 


        
        
    def plot_pca_cluster(self,data, pc = 0, gap = 1, density = 20, lw = 1, title = None):
        f, axs = plt.subplots(self.n_ver, self.n_hor, figsize=(self.fig_ver, self.fig_hor), facecolor='w', edgecolor='k')
        f.suptitle(title)
        pc_idx = pc+1
        for ax in axs.flatten():
            ax.scatter(data[:,pc][::density],data[:, pc_idx % 8][::density], marker = '.', s = lw, color = self.color)
            ax.title.set_text( f'{pc} vs {pc_idx% 8 }')
            pc_idx += gap
        return None
    
    def plot_stream_cluster(self,data, pc = 0, gap = 1, density = True, lw = 1, title = None):
        f, axs = plt.subplots(self.n_ver, self.n_hor, figsize=(self.fig_ver, self.fig_hor), facecolor='w', edgecolor='k')
        f.suptitle(title)
        pc_idx = pc+1
        for ax in axs.flatten():
            ax.hist2d(data[:,pc][::density],data[:, pc_idx % 8][::density], density = density, norm = LogNorm())
            ax.title.set_text( f'{pc} vs {pc_idx% 8 }')
            pc_idx += gap
        return None
    
    def plot_masked(self,data, masked_idx = 0, gap = 1, title = None):
        f, axs = plt.subplots(self.n_ver, self.n_hor, figsize=(self.fig_ver, self.fig_hor), dpi=120, facecolor='w', edgecolor='k')
        f.suptitle(title)
        pixel = self.ver* self.hor
        for ax in axs.flatten():
            ax.matshow(np.reshape(data[pixel* masked_idx : pixel*(masked_idx+1)],(self.ver,self.hor)))
            ax.set_title(f'{masked_idx+1}')
            ax.axis('off')
            masked_idx += gap
        return None

    def plot_masked_stream(self,data, masked_idx = 0, gap = 1, title = None):
        f, axs = plt.subplots(5, 8, figsize=(self.fig_ver, self.fig_hor), dpi=120, facecolor='w', edgecolor='k')
        f.suptitle(title)
        pixel = self.ver* self.hor
        for ax in axs.flatten():
            ax.matshow(np.reshape(data[pixel* masked_idx : pixel*(masked_idx+1)],(self.ver,self.hor)))
            ax.set_title(f'{masked_idx+1}')
            ax.axis('off')
            masked_idx += gap
        return None
    
    def get_intensity_cutoff(self, intensity, cutoff = 750):
        pixel = self.ver * self.hor
        f,axs = plt.subplots(4,10, figsize = (20,12))
        idx = 0
        for ax in axs.flatten():
            image = np.reshape(intensity[pixel*idx: pixel*(idx+1)], (self.ver , self.hor))
            ax.matshow(image> cutoff)
            ax.axis('off')
            idx += 1

    
# =============================================================================
#     
#     def plot_k_layer(self):
#         cc_new = Counter(self.cc.merge)
#         most_common = cc_new.most_common(self.cc.topk)
#         merge_k = int(max(self.cc.merge)+1)
#         print(most_common)
#         f, axs = plt.subplots(self.n_ver, self.n_hor, figsize=(self.fig_ver, self.fig_hor), dpi=120, facecolor='w', edgecolor='k')
#         layer_idx = 0
#         for ax in axs.flatten():
#             merge_color_k = self.cc.colormap.copy()
#             merge_color_k[self.cc.merge != most_common[layer_idx][0]] = -1
#             merge_reshape = np.reshape(merge_color_k, [self.cc.ver, self.cc.hor])
#             ax.imshow(merge_reshape)
#             ax.title.set_text(f'top {layer_idx+1}, base {self.cc.base}')
#             layer_idx += 1
#         return None
# =============================================================================
    
    def plot_topk_color(self, option, cc):
        if option == 'sort':
            colormap = cc.color_map_sort
        elif option == 'sketch':
            colormap = cc.color_map_sketch
        else:
            return ('select option in sort and sketch')
        f, axs = plt.subplots(self.n_ver, self.n_hor, figsize=(self.fig_ver, self.fig_hor), dpi=120, facecolor='w', edgecolor='k')
        f.suptitle(f'Count {option} Topk', fontsize=self.txt_size)
        layer_idx = 1
        for ax in axs.flatten():
            merge_color_k = colormap.copy()
            merge_color_k[colormap != layer_idx] = 0
            merge_reshape = np.reshape(merge_color_k, [cc.ver, cc.hor])
            ax.imshow(merge_reshape)
            ax.title.set_text(f'Image {self.idx}: top {layer_idx} color, base {cc.base}' )
#            ax.titlesize = 5
            ax.labelsize = 5
            layer_idx += 1
        return None
    
    def plot_topk_diff(self, cc):
        colormap = cc.color_map_sort - cc.color_map_sketch
        error = round( np.sum(colormap!=0)/cc.pixel * 100, 2)
        
        print("Error:" ,error)
        merge_reshape = np.reshape(colormap, [cc.ver, cc.hor])
        f, ax = plt.subplots( figsize=(5, self.fig_hor), dpi=120, facecolor='w', edgecolor='k')
        ax.imshow(merge_reshape)
        ax.title.set_text(f'Image {self.idx}: {error} % Error ' )
        return None
    
def topk_masked(stream_concat, HHvals, N_masked = 41, topk = 50, bg = 10):
    pixel = images.ver*images.hor
    stream_1Ds = stream_concat[:N_masked*pixel]
    stream_masked = stream_1Ds
    for idx, hhval in enumerate( HHvals): 
    #     print (hhval,idx)
        stream_masked = np.where(stream_1Ds != hhval, stream_masked,  -1-idx ) 

        if idx > topk:
            break
    stream_masked = np.where(stream_masked < 0, stream_masked,  bg ) 
    # stream_masked = abs(stream_masked)
    print(stream_masked,np.shape(stream_masked))
    return stream_masked

    
def justplot(data, title = None):
    plt.figure(dpi = 1000)
    plt.matshow(np.reshape(data, (1004, 1344)))
    plt.title(title)