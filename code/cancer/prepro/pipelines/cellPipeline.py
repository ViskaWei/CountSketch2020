import logging
import os
import time
import numpy as np
import pickle

from cancer.scripts.script import Script
from cancer.prepro.data.cellDataset import CellDataset
from cancer.prepro.data.bulk import Bulk
from util.prepro import get_encode_stream
from util.HH import get_HH_pd
from cancer.postpro.project import get_umap_pd,get_kmean_lbl,get_pred_stream
 

class CellPipeline(Script):
    def __init__(self, logging=True):
        super().__init__()
        self.nImg=None
        self.dim=None
        self.smooth=None
        self.cutoff=None
        self.base=None
        self.dtype='uint64'
        self.save={'mat': False, 'mask':False, 'stream':False, 'HHs':False, 'maskId':None}
        self.idx=None
        self.sketchMode='exact'
        
    def add_args(self, parser):
        super().add_args(parser)
        # ===========================  LOAD  ================================
        parser.add_argument('--nImg', type=int, help='num of image loading\n')
        parser.add_argument('--test', type=bool, help='Test or original size\n')
        parser.add_argument('--smooth', type=float, default=None, help='Gaussian smooth sigma\n')
        parser.add_argument('--saveMat', type=bool, help='Saving mat\n')
        parser.add_argument('--saveMask', type=bool, help='Saving mask\n')
        parser.add_argument('--saveStream', type=bool, help='Saving stream\n')
        parser.add_argument('--saveHHs', type=bool, help='Saving HH\n')

        parser.add_argument('--sketchMode', type=str, help='exact or cs\n')


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
        self.apply_sketch_args()
        self.apply_save_args()

    def apply_dataset_args(self):
        if 'in' not in self.args or self.args['in'] is None:
            raise "--in input directory is not specified"

        if 'nImg' in self.args and self.args['nImg'] is not None:
            self.nImg=self.args['nImg']
        
        if 'smooth' in self.args and self.args['smooth'] is not None:
            self.smooth=[self.args['smooth'],self.args['smooth'],0]

    def apply_prepro_args(self):
        if 'cutoff' in self.args and self.args['cutoff'] is not None:
            try:
                self.cutoff=pickle.load(open(self.args['cutoff'],'rb')) 
            except:
                self.cutoff=None
                logging.info('cannot load cutoff, calculating again')
    
    def apply_encode_args(self):
        if 'base' in self.args and self.args['base'] is not None:
            self.base=self.args['base']
        else:
            raise "--base base not specified"
        if 'dtype' in self.args and self.args['dtype'] is not None:
            self.dtype=self.args['dtype']

    def apply_sketch_args(self):
        if 'sketchMode' in self.args and self.args['sketchMode'] is not None:
            self.sketchMode=self.args['sketchMode']
        
    def apply_save_args(self):
        if 'saveMat' in self.args and self.args['saveMat'] is not None:
            self.save['mat']=self.args['saveMat']
        if 'saveMask' in self.args and self.args['saveMask'] is not None:
            self.save['mask']=self.args['saveMask']
        if 'saveStream' in self.args and self.args['saveStream'] is not None:
            self.save['stream']=self.args['saveStream']
        if 'saveHHs' in self.args and self.args['saveHHs'] is not None:
            self.save['HHs']=self.args['saveHHs']
        if self.save['mask']:
            if 'maskId' in self.args and self.args['maskId'] is not None:
                self.save['maskId'] = self.args['maskId']
        logging.info('saving {}'.format(self.save.items()))
        

    def run(self):
        mat = self.run_step_load()
        df_norm=self.run_step_norm(mat)
        stream=self.run_step_encode(df_norm)
        HHs = self.run_step_sketch(stream)
        self.run_step_save()


    def run_step_load(self):
        ds=CellDataset(self.args['in'] ,self.nImg)
        img_loader=ds.get_img_loader(self.args['test'], smooth=self.smooth)  
        ds.load(img_loader, True)
        self.nImg=ds.nImg
        if self.save['maskId'] is not None:
            if self.save['maskId'] >self.nImg:
                self.save['maskId'] = 0
                logging.info('maskId out of range, saving 0th img')

        ds.get_pc(self.dim)
        mat = ds.get_bulk()
        del ds
        if self.save['mat']: self.save_txt(mat, 'mat')        
        return mat  

    def run_step_save(self):
        pass

    def save_txt(self, mat, filename):
        name=f'{self.out}/{filename}.txt'
        logging.info('  saving {}'.format(name))
        np.savetxt( name, mat)
    
    def save_mask(self,mask, filename):
        mask2d=mask.reshape((self.nImg,1004*1344))
        if self.save['maskId'] is None:
            name=f'{self.out}/{filename}_all.txt' 
            logging.info('  saving {}'.format(name))
            np.savetxt(name, mask)
        else:
            maskId=self.save['maskId']
            mask0= mask2d[maskId]
            idxi=int(mask2d[:maskId].sum())
            idxj=int(mask2d[:(maskId+1)].sum())
            assert idxj-idxi == mask0.sum()
            self.idx=[idxi,idxj,maskId]
            logging.info('idxi, idxj, maskId: {}'.format(self.idx))
            logging.info('  saving mask {}{}'.format(mask0.shape, mask.sum()))
            np.savetxt(f'{self.out}/{filename}Id{maskId}.txt' , mask0)  

    
    def run_step_norm(self, mat):
        bulk=Bulk(mat)
        assert bulk.dim == self.dim
        intensity=bulk.get_intensity()
        if self.cutoff is None:
            self.cutoff = bulk.get_cutoff(intensity)
            pickle.dump(self.cutoff,open(f'{self.out}/cutoff.txt','wb'))
        logging.info(" cutoff @:  {}".format(self.cutoff))
        df_norm, mask = bulk.get_unit_ball(intensity, self.cutoff)
        del bulk
        if self.save['mask']: self.save_mask(mask,'mask')
        return df_norm

    def run_step_encode(self, df_norm):
        stream=get_encode_stream(df_norm, self.base, self.dtype)
        if self.save['stream']: 
            self.save_txt(stream, 'stream')
        elif self.idx is not None:
            self.save_txt(stream[self.idx[0]:self.idx[1]],f'stream{self.idx[-1]}')
        return stream
    
    def run_step_sketch(self, stream):
        if self.sketchMode=='exact':
            HH_pd=get_HH_pd(stream,self.base,self.dim, self.dtype, True, None)
        else:
            raise 'exact only now'
            # HH_pd=get_HH_pd(stream,base,ftr_len, dtype, False, topk, r=16, d=1000000,c=None,device=None)
        if self.save['HHs']:   
            HH_pd.to_csv(f'{self.out}/HH_pd_b{self.base}_{self.sketchMode}.csv',index=False)
        return HH_pd


    
   

 