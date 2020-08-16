import logging
import os
import time
import numpy as np

from cancer.prepro.data.cellDataset import CellDataset
from cancer.scripts.script import Script

class CellPipeline(Script):
    def __init__(self, logging=True):
        super().__init__()
        self.dim=None
        self.vecs=None
        
    def add_args(self, parser):
        super().add_args(parser)
        # ===========================  LOAD  ================================
        parser.add_argument('--isTest', action='store_true', help='Correct for extinction\n')
        parser.add_argument('--isSmooth', action='store_true', help='Correct for extinction\n')
        parser.add_argument('--smooth-sig', type=float, default=2.0, help='Wavelength range\n')
        parser.add_argument('--dim', type=int, default=8,  help='Latent Representation dimension\n')

        # ===========================  PREPRO  ================================
        parser.add_argument('--', type=str, default=None, help='Broadband filter for magnitude calculation.\n')
        
    def prepare(self):
        super().prepare()
        print('prepare')
        self.apply_dataset_args()

    def apply_dataset_args(self):
        if 'in' in self.args and self.args['in'] is not None:
            ds=CellDataset(self.args['in'] ,self.args['nImg'])
            smooth_sig=[self.args['smooth_sig'],self.args['smooth_sig'],0]
            loader_fn=ds.loader(self.args['isTest'],self.args['isSmooth'], smooth_sig)  
            ds.load(loader_fn, True)
        else:
            raise "--in input directory is not specified"
        
        if 'dim' in self.args and self.args['dim'] is not None:
            self.dim = self.args["dim"]
            ds.get_pc(self.dim)
            self.vecs = ds.get_bulk()
        else:
            raise "--dim latent dimension not specified"
            
    def apply_prepro_args(self):


 