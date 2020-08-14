from multiprocessing import Pool, cpu_count, current_process
import math
import sys
sys.path.insert(0, "/home/ivkinnikita/astro6d/py_halo/utils/")
import binreader as br
import numpy as np
import pprint as pp
import torch


"""This is a class to get exacct frequency counts for 4D meshed particle data"""
class exact_hh:
        box_size = 500      
        vel_size = 20

        """
        Initializes an exact heavy hitter finding object. 
        """
        def __init__(self,p):
                self.exact_counts = np.zeros(self.vel_size*(self.box_size**3),
                                                dtype=np.int32)
                self.reader = br.binreader()
                self.data = None
                self.device = 'cuda'
                self.p = p

        """
        Iterates through all particle information and gets counts for every
        single cell 
        """
        def count(self):
            num_parts = 10000000
            self.data = self.reader.process(num_parts)
            while (len(self.data) > 0):
                args0 = torch.tensor(np.array([self.data[i*6] for i in\
                                             range(len(self.data)//6)]),\
                                     device=self.device).long()
                args1 = torch.tensor(np.array([self.data[i*6+1] for i in\
                                             range(len(self.data)//6)]),\
                                    device=self.device).long()
                args2 = torch.tensor(np.array([self.data[i*6+2] for i in\
                                             range(len(self.data)//6)]),\
                                    device=self.device).long()
                args3 = torch.tensor(np.array([self.data[i*6+(self.p)] for i in\
                                             range(len(self.data)//6)]),\
                                    device=self.device).long()
                

                args1.mul_(500)
                args2.mul_(500*500)
                
                # shift relevant particles into positives
                args3.add_(2000)

                # break 0-4000 velocities into 0 - 20 boxes
                args3.div_(200) 
                                
                keys = (args0.add(args1)).add(args2)
                keys.mul_(20)
                keys.add_(args3)
                # get rid of overly high or low velocities send them all to the
                # cell with key_id = 0
                zeros = torch.zeros(len(args3), device=self.device).long()
                keys = torch.where((args3 < 0), zeros, keys)
                keys = torch.where((args3 > 4000), zeros, keys)
                self.exact_counts = np.add(self.exact_counts,
                                           np.bincount(keys.cpu().numpy(),\
                                                    minlength=20*(500**3)))

                self.data = self.reader.process(num_parts)

                      
 
if __name__ == "__main__":
        hh = exact_hh(3)
        hh.count()
        out = open("/srv/scratch1/millennium/exact_cells/new_xvel_20bin", "w")
        for cell in hh.exact_counts:
            out.write(str(cell) + '\n')
        out.close()
 
        hh = exact_hh(4)
        hh.count()
        out = open("/srv/scratch1/millennium/exact_cells/new_yvel_20bin", "w")
        for cell in hh.exact_counts:
            out.write(str(cell) + '\n')
        out.close()
        
        hh = exact_hh(5)
        hh.count()
        out = open("/srv/scratch1/millennium/exact_cells/new_zvel_20bin", "w")
        for cell in hh.exact_counts:
            out.write(str(cell) + '\n')
        out.close()


