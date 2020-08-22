import math
import numpy as np
import copy
import torch
LARGEPRIME = 2**61-1
# torch.random.manual_seed(42)
class CSVec(object):
    def __init__(self, d, c, r, k, device=None):
        self.r = r # num of rows
        self.c = c # num of columns
        self.d = int(d) # vector dimensionality
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            if (not isinstance(device, torch.device) and
                    not ("cuda" in device or device == "cpu")):
                msg = "Expected a valid device, got {}"
                raise ValueError(msg.format(device))
        self.device = device
        self.table = torch.zeros((r, c), device=self.device)

        torch.random.manual_seed(42)
        rand_state = torch.random.get_rng_state()
        self.hashes = torch.randint(0, LARGEPRIME, (self.r, 6),
                               dtype=torch.int64, device="cpu")
        torch.random.set_rng_state(rand_state)
        self.h1 = self.hashes[:,0:1]
        self.h2 = self.hashes[:,1:2]
        self.h3 = self.hashes[:,2:3]
        self.h4 = self.hashes[:,3:4]
        self.h5 = self.hashes[:,4:5]
        self.h6 = self.hashes[:,5:6]

        self.topk = torch.zeros((k,2), dtype=torch.int64, device=self.device)        
        
#     def accumulateVec(self, vec1, vec2):
#         vec1 = vec1.to(self.device)
#         vec2 = vec2.to(self.device)

#         assert(len(vec.size()) == 1)
#         signs = (((self.h1 * vec + self.h2) * vec + self.h3) * vec + self.h4)
#         signs = ((signs % LARGEPRIME % 2) * 2 - 1).float()
#         signs = signs.to(self.device)

#         # computing bucket hashes (2-wise independence)

#         buckets = ((self.h5 * vec) + self.h6) % LARGEPRIME % self.c
#         buckets = buckets.to(self.device)

#         # the vector is sketched to each row independently
#         for r in range(self.r):
#             bucket = buckets[r,:]
#             sign = signs[r,:]
#             # print('bucket', r, bucket, sign)
#             self.table[r,:] += torch.bincount(input=bucket,
#                                               weights=sign,
#                                               minlength=self.c)
            
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

    def accumulateVec(self, vec):
        assert(len(vec.size()) == 1)
        signs = (((self.h1 * vec + self.h2) * vec + self.h3) * vec + self.h4)
        signs = ((signs % LARGEPRIME % 2) * 2 - 1).float()
        signs = signs.to(self.device)
        # computing bucket hashes (2-wise independence)
        buckets = ((self.h5 * vec) + self.h6) % LARGEPRIME % self.c
        buckets = buckets.to(self.device)
        for r in range(self.r):
            bucket = buckets[r,:]
            sign = signs[r,:]
            self.table[r,:] += torch.bincount(input=bucket,
                                                weights=sign,
                                                minlength=self.c)
    def query(self, vec):
        signs = (((self.h1 * vec + self.h2) * vec + self.h3) * vec + self.h4)
        signs = ((signs % LARGEPRIME % 2) * 2 - 1).float()
        signs = signs.to(self.device)
        # computing bucket hashes (2-wise independence)
        buckets = ((self.h5 * vec) + self.h6) % LARGEPRIME % self.c
        buckets = buckets.to(self.device)
        vals = torch.zeros(self.r, vec.size()[0],dtype=torch.int64, device=self.device)#
        for r in range(self.r):
            bucket = buckets[r,:]
            sign = signs[r,:]           
            vals[r] = self.table[r, bucket] * sign
        return vals.median(dim=0)[0]

