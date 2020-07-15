import math
import numpy as np
import copy
import torch
LARGEPRIME = 2**61-1
class CSVec(object):
    """ Count Sketch of a vector
    Treating a vector as a stream of tokens with associated weights,
    this class computes the count sketch of an input vector, and
    supports operations on the resulting sketch.
    """

    def __init__(self, d, c, r, k, device=None):
        """ Constductor for CSVec
        Args:
            d: the cardinality of the skteched vector
            c: the number of columns (buckets) in the sketch
            r: the number of rows in the sketch
            k: the number of Heavy Hitters
            device: which device to use (cuda or cpu). If None, chooses
                cuda if available, else cpu
        """
        self.r = r # num of rows
        self.c = c # num of columns
        # need int() here b/c annoying np returning np.int64...
        self.d = int(d) # vector dimensionality
        # choose the device automatically if none was given
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            if (not isinstance(device, torch.device) and
                    not ("cuda" in device or device == "cpu")):
                msg = "Expected a valid device, got {}"
                raise ValueError(msg.format(device))

        self.device = device

        # initialize the sketch to all zeros
        self.table = torch.zeros((r, c), device=self.device)

        # initialize hashing functions for each row:
        # 2 random numbers for bucket hashes + 4 random numbers for
        # sign hashes
        # maintain existing random state so we don't mess with
        # the main module trying to set the random seed but still
        # get reproducible hashes for the same value of r

        # do all these computations on the CPU, since pytorch
        # is incapable of in-place mod, and without that, this
        # computation uses up too much GPU RAM
        rand_state = torch.random.get_rng_state()
        torch.random.manual_seed(42)
        self.hashes = torch.randint(0, LARGEPRIME, (self.r, 6),
                               dtype=torch.int64, device="cpu")
        torch.random.set_rng_state(rand_state)
        # computing sign hashes (4 wise independence)
        self.h1 = self.hashes[:,0:1]
        self.h2 = self.hashes[:,1:2]
        self.h3 = self.hashes[:,2:3]
        self.h4 = self.hashes[:,3:4]
        self.h5 = self.hashes[:,4:5]
        self.h6 = self.hashes[:,5:6]

        self.top_k = torch.zeros((k,2), dtype=torch.int64, device=self.device)        
        self.k = k
        
    def accumulateVec(self, vec):
        """ Sketches a vector and adds the result to self
        Args:
            vec: the vector to be sketched
        """
#         vec = vec.to(self.device)

        assert(len(vec.size()) == 1)
        signs = (((self.h1 * vec + self.h2) * vec + self.h3) * vec + self.h4)
        signs = ((signs % LARGEPRIME % 2) * 2 - 1).float()
        signs = signs.to(self.device)

        # computing bucket hashes (2-wise independence)

        buckets = ((self.h5 * vec) + self.h6) % LARGEPRIME % self.c
        buckets = buckets.to(self.device)

        # the vector is sketched to each row independently
        for r in range(self.r):
            bucket = buckets[r,:]
            sign = signs[r,:]
            # print('bucket', r, bucket, sign)
            self.table[r,:] += torch.bincount(input=bucket,
                                              weights=sign,
                                              minlength=self.c)
            
    def _findValues(self, vec):
        # computing sign hashes (4 wise independence)
        signs = (((self.h1 * vec + self.h2) * vec + self.h3) * vec + self.h4)
        signs = ((signs % LARGEPRIME % 2) * 2 - 1).float()
        signs = signs.to(self.device)

        # computing bucket hashes (2-wise independence)
        buckets = ((self.h5 * vec) + self.h6) % LARGEPRIME % self.c
        buckets = buckets.to(self.device)  
        # estimating frequency of input coordinates
        d = vec.size()[0]
        vals = torch.zeros(self.r, d, device=self.device)
        for r in range(self.r):
            vals[r] = self.table[r, buckets[r]] * signs[r]
        return vals.median(dim=0)[0]

    def get_CS_HHs_dict(self, base, pca_comp):
        '''
        use this when frequency vector is smaller than stream_1d 
        '''
        val_dict = torch.arange(base**pca_comp, dtype=torch.int64)
        freqs = self._findValues(val_dict)
        ############## sorting frequency vector ###############
        cs_freq, cs_val = torch.sort(freqs, descending = True)
        
        cs_freq = cs_freq.cpu().numpy()
        cs_val = cs_val.cpu().numpy()
        return cs_freq, cs_val
    
    def get_CS_HHs_unique(self, stream_tr):
        '''
        use this when frequency vector is larger than stream_1d
        '''
        val_dict = torch.unique(stream_tr)
        freqs = self._findValues(val_dict)
        cs_freq, cs_val = torch.sort(freqs, descending = True)
        cs_freq, cs_val = cs_freq[:self.k], val_dict[cs_val[:self.k]]
        cs_freq = cs_freq.cpu().numpy()
        cs_val = cs_val.cpu().numpy()
        return cs_freq, cs_val
    
    def get_CS_table(self, stream_tr):
        vecs = stream_tr
        while len(vecs) > 0:
            vec = vecs[:self.d] 
            self.accumulateVec(vec)    
            vecs = vecs[self.d:]
        return None

    def accumulateVec_heap(self, vec, min_freq = 2000):
        assert(len(vec.size()) == 1)
        signs = (((self.h1 * vec + self.h2) * vec + self.h3) * vec + self.h4)
        signs = ((signs % LARGEPRIME % 2) * 2 - 1).float()
        signs = signs.to(self.device)
        # computing bucket hashes (2-wise independence)
        buckets = ((self.h5 * vec) + self.h6) % LARGEPRIME % self.c
        buckets = buckets.to(self.device)
        vals = torch.zeros(self.r, vec.size()[0],dtype=torch.int64, device=self.device)#
        # the vector is sketched to each row independently
        for r in range(self.r):
            bucket = buckets[r,:]
            sign = signs[r,:]
            self.table[r,:] += torch.bincount(input=bucket,
                                                weights=sign,
                                                minlength=self.c)
            vals[r] = self.table[r, bucket] * sign
        vals = vals.median(dim=0)[0]# this is their estimated freq
        # keeping a heap
        coord = vec.to(self.device)
        vals = torch.stack((vals, coord), dim=1)
        print(vals.shape)
        heap_min = torch.tensor([self.top_k[-1,0], min_freq]).max()
        for val in vals:
            if val[0] > heap_min: 
                if val[1] not in self.top_k[:,1]:
                    hhvals = self.top_k[:,0]
                    cutoff = torch.argmin(hhvals)
                    self.top_k[cutoff] = val
                    heap_min = self.top_k[:,0].min()
        self.top_k = self.top_k[torch.argsort(self.top_k[:,0], 0, descending=True)]
        
    def l2estimate(self):
        """ Return an estimate of the L2 norm of the sketch """
        # l2 norm esimation from the sketch
        return np.sqrt(torch.median(torch.sum(self.table**2,1)).item())
      
    def zero(self):
        """ Set all the entries of the sketch to zero """
        self.table.zero_()

    def cpu_(self):
        self.device = "cpu"
        self.table = self.table.cpu()

    def cuda_(self, device="cuda"):
        self.device = device
        self.table = self.table.cuda()
    
    # ## try out with and wihtout 
    # def half_(self):
    #     self.table = self.table.half()

    # def float_(self):
    #     self.table = self.table.float()



# def run_step_vec(stream_1D, base, pca_comp):
#     assert(np.allclose(stream_1D, stream_1D.astype("Int64")))
#     stream_1D = stream_1D.astype("Int64")
#     l_stream_1D = len(stream_1D)
#     print('stream_1D O(space)',l_stream_1D, np.log(l_stream_1D).round(2), np.log10(l_stream_1D).round(2))
#     l_vec = base**pca_comp
#     vec = np.zeros(l_vec)
#     print('vec O(space):',l_vec, base, pca_comp, np.log10(l_vec).round(2))
#     for val in stream_1D:
#         vec[val] += 1
#     vec_tr = torch.from_numpy(np.array(vec)).float().to("cuda")
#     return vec_tr


# def run_step_sketch(vec_tr, d, topk, col, row):
#     cs = CSVec(d = d , c=col, r=row)
#     cs.accumulateVec(vec_tr)
#     HH = cs._findHHK_v(topk)
#     val = HH[0].cpu().numpy()
#     freq = HH[1].cpu().numpy()
#     return val,freq


# def process_countsketch(d, stream_1D, base, pca_comp, topk, col_range, row_range):
#     vec_tr = run_step_vec(stream_1D, base, pca_comp)
#     sketchs = {}
#     for row in row_range:     
#         for col in col_range:
#             val,freq = run_step_sketch(vec_tr, d,topk, col,row)
#             sketchs[f'{row}_{col}_val'] = val
#             sketchs[f'{row}_{col}_freq'] = freq
#     return val,freq

# def get_CS_HHs_batch(self, val_dicts0, min_freq = 1000):
#     cs_vals = torch.tensor([], device = device).type(torch.LongTensor)
#     cs_freqs =  torch.tensor([], device = device).type(torch.FloatTensor)
#     while len(val_dicts0) > 0:
#         val_dict = val_dicts0[:self.d]
#         freqs = csv._findValues(val_dict)
#         mask = (freqs > min_freq)
#         val_dict, freqs = val_dict[mask], freqs[mask]
#         cs_freq, cs_val = torch.sort(freqs, descending = True)
#         cs_freqs = torch.cat((cs_freqs, cs_freq[:self.k]))
#         cs_vals = torch.cat((cs_vals, val_dict[cs_val[:self.k]]))
#         val_dicts0 = val_dicts0[self.d:]
#         print(len(val_dicts0))

#     cs_freqs_sorted, cs_vals_idx = torch.sort(cs_freqs, descending = True)
#     cs_freqs = cs_freqs_sorted[:self.k]
#     cs_vals = cs_vals[cs_vals_idx][:self.k]
#     return cs_freqs, cs_vals

# cs_freq1, cs_val1 = torch.sort( csv._findValues(val_dicts0), descending = True)
# assert torch.sum(cs_freq1[:len(cs_freqs)] != cs_freqs) == 0
# assert torch.sum(cs_val1[:len(cs_vals)] != cs_vals) == 0