import logging
import os
import time
import numpy as np
import pickle

from cancer.scripts.script import Script
from cancer.prepro.data.cellDataset import CellDataset
from cancer.prepro.data.bulk import Bulk


class CellPipeline(Script):
    def __init__(self, logging=True):
        super().__init__()
        self.dim=None
        self.vecs=None
        self.smooth=None
        
    def add_args(self, parser):
        super().add_args(parser)
        # ===========================  LOAD  ================================
        parser.add_argument('--test', type=bool, help='Test or original size\n')
        parser.add_argument('--smooth', type=float, default=None, help='Gaussian smooth sigma\n')

        # ===========================  PREPRO  ================================
        parser.add_argument('--save', type=bool, help='Saving vecs\n')

        # parser.add_argument('--', type=str, default=None, help='Broadband filter for magnitude calculation.\n')
        
    def prepare(self):
        super().prepare()
        self.apply_dataset_args()
        self.apply_prepro_args()

    def apply_dataset_args(self):
        if 'in' not in self.args or self.args['in'] is None:
            raise "--in input directory is not specified"
        
        if 'smooth' in self.args and self.args['smooth'] is not None:
            self.smooth=[self.args['smooth'],self.args['smooth'],0]

    def apply_prepro_args(self):
        if 'cutoff' in self.args and self.args['cutoff'] is not None:
            self.cutoff=pickle.load(open(self.args['cutoff'],'rb')) 
        

    def run(self):
        self.run_step_load()
        self.run_step_save()


    def run_step_load(self):
        ds=CellDataset(self.args['in'] ,self.args['nImg'])
        img_loader=ds.get_img_loader(self.args['test'], smooth=self.smooth)  
        ds.load(img_loader, True)
        ds.get_pc(self.dim)
        self.vecs = ds.get_bulk()  

    def run_step_save(self):
        if self.args['save']:  
            np.savetxt( self.name, self.vecs)

    def run_step_ball(self):
        bulk=Bulk(self.vecs)
        intensity=get_intensity()
        if self.cutoff is None:
            self.cutoff = bulk.get_cutoff(intensity)
            pickle.dump(cut,open(f'{self.out}/cutoff.txt','wb'))
        logging.info(" cutoff @:  {}".format(self.cutoff))

        self.get_unit_ball(self.cutoff)


    
        
   

 