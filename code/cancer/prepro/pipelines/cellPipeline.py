import logging
import os
import time
import numpy as np
import pickle

from cancer.scripts.script import Script
from cancer.prepro.data.cellDataset import CellDataset
from cancer.prepro.data.bulk import Bulk
from util.prepro import get_encode_stream


class CellPipeline(Script):
    def __init__(self, logging=True):
        super().__init__()
        self.dim=None
        self.vecs=None
        self.smooth=None
        self.cutoff=None
        self.base=None
        self.dtype='uint64'
        self.maskId=26
        
    def add_args(self, parser):
        super().add_args(parser)
        # ===========================  LOAD  ================================
        parser.add_argument('--nImg', type=int, help='num of image loading\n')
        parser.add_argument('--test', type=bool, help='Test or original size\n')
        parser.add_argument('--smooth', type=float, default=None, help='Gaussian smooth sigma\n')
        parser.add_argument('--save', type=bool, help='Saving vecs\n')
        parser.add_argument('--maskId', type=int, help='Id of mask saved\n')


        # ===========================  PREPRO  ================================
        parser.add_argument('--cutoff', type=str, default=None, help='Bg cutoff\n')
        parser.add_argument('--base', type=int, default=None, help='Base\n')
        parser.add_argument('--dtype', type=str, default=None, help='dtype\n')


        
    def prepare(self):
        super().prepare()
        self.apply_dataset_args()
        self.apply_prepro_args()
        self.apply_encode_args()

    def apply_dataset_args(self):
        if 'in' not in self.args or self.args['in'] is None:
            raise "--in input directory is not specified"
        
        if 'smooth' in self.args and self.args['smooth'] is not None:
            self.smooth=[self.args['smooth'],self.args['smooth'],0]

    def apply_prepro_args(self):
        if 'cutoff' in self.args and self.args['cutoff'] is not None:
            self.cutoff=pickle.load(open(self.args['cutoff'],'rb')) 
    
    def apply_encode_args(self):
        if 'base' in self.args and self.args['base'] is not None:
            self.base=self.args['base']
        else:
            raise "--base base not specified"
        if 'dtype' in self.args and self.args['dtype'] is not None:
            self.dtype=self.args['dtype']
        

    def run(self):
        self.run_step_load()
        stream1D =self.run_step_encode()
        if self.args['save']:  
            self.run_step_save()

    def run_step_load(self):
        ds=CellDataset(self.args['in'] ,self.args['nImg'])
        img_loader=ds.get_img_loader(self.args['test'], smooth=self.smooth)  
        ds.load(img_loader, True)
        ds.get_pc(self.dim)
        self.vecs = ds.get_bulk()  

    def run_step_save(self):
        if self.maskId is not None:
            mask2d=mask.reshape((self.nImg,1004*1344))
            # np.savetxt( f'{self.out}/{self.name}BulkVecs.txt', self.vecs)
            
    def run_step_encode(self):
        bulk=Bulk(self.vecs)
        assert bulk.dim == self.dim
        intensity=bulk.get_intensity()
        if self.cutoff is None:
            self.cutoff = bulk.get_cutoff(intensity)
            pickle.dump(self.cutoff,open(f'{self.out}/cutoff.txt','wb'))
        logging.info(" cutoff @:  {}".format(self.cutoff))
        df_norm, mask = bulk.get_unit_ball(intensity, self.cutoff)
        del bulk
        stream1D=get_encode_stream(df_norm, self.base, self.dtype)
        return stream1D, mask
    
    def run_step_sketch(self):
        pass

        
   

 