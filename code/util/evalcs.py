
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from pylab import *



class Evalsketch():
    def __init__(self, exact_pd, cs_pd, base, pca_comp, min_freq=None):
        self.cs_freq = cs_pd['freq'].values
        self.cs_HH = cs_pd['HH'].values
        self.cs_rk=cs_pd['freq'].cumsum().values
        self.base = base
        self.pca_comp = pca_comp
        # self.pixel = pixel
        if min_freq is not None:
            exact_pd = exact_pd.loc[exact_pd['freq']>min_freq]
        self.full = exact_pd.shape[0]
        self.k=len(self.cs_freq)
        print("Topk is", self.k)
        self.exact_freq = exact_pd['freq'].values
        self.exact_HH = exact_pd['HH'].values
        self.exact_rk=exact_pd['freq'].cumsum().values
        
    def plot_all(self):
        f, (ax0,ax1)=plt.subplots(2,1,figsize=(20,6))
        self.plot_log_hist(ax=ax0)
        error=self.plot_rel_error(ax=ax1)
        return error
        # Plotting
#         self.plot_log_hist()
        
#         self.plot_rel_error()

    def plot_log_hist(self, bins = 20, ax = None):
        if ax is None:
            ax = plt.gca()
        plt.style.use('seaborn-deep')
        x = np.log10(self.exact_freq)
        y = np.log10(self.cs_freq)
        ax.hist([x, y], log = True, bins = 50,label=['Exact', 'Sketch'])
        ax.legend(loc='upper right')
        ax.set_xlabel('counts in log10')
        ax.set_ylabel('Number of HHs')
    
#     def plot_rank(self, bins = 20, ax = None):
#         if ax is None:
#             ax = plt.gca()
#         plt.style.use('seaborn-deep')
#         x = np.log10(self.exact_rk)
#         y = np.log10(self.cs_rk)
#         ax.plot(x, log = True, bins = 50,label=['Exact', 'Sketch'])
#         ax.legend(loc='upper right')
#         ax.set_xlabel('counts in log10')
#         ax.set_ylabel('rank of HHs')
        
    def get_relative_error(self):
        error = np.ones(self.k)
        for key, val in enumerate(self.exact_HH): 
            if val in self.cs_HH:
                c0 = self.exact_freq[key]
                c1 = self.cs_freq[self.cs_HH == val]
                # error[key] -= np.min([c1/c0,2])
                error[key] -= c1/c0
        return error  
         
    def plot_rel_error(self, ax = None):
        if ax is None:
            ax = plt.gca()
        error = self.get_relative_error()
        ax.scatter(np.arange(self.k), error, s = 1, label = f"b={self.base}, topk=20k")
        ax.set_xlabel('Rank')
        ax.set_ylabel('Relative Error')
        ax.set_ylim([-0.5,0.5])
        ax.set_xlim([0,self.k])
        ax.legend(loc = 1)
        return error
                  

# plt.figure(figsize=(10,6))
# plt.plot(np.log(HH_pd.index)/np.log(2.0),np.log(HH_pd['freq'])/np.log(2.0))
# plt.axvline(np.log(20000)/np.log(2.0),c='r',linestyle=':', label='Topk=20000')
# plt.ylabel('log2(freq)')
# plt.xlabel('log2(rank)')
# plt.xlim(0,18)
# plt.legend()
# plt.grid()
# plt.savefig('freq_vs_rank.png')
# plt.figure(figsize=(8,6))
# plt.plot(HH_CSh['rk']/26034006.0)
# plt.xlabel('rank')
# plt.ylabel('cumulative fraction')
# plt.ylim(0,1)
# plt.grid()
# plt.savefig('cumulative_frac.png')