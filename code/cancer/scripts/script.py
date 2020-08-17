import os
import sys
import json
import logging
import argparse
import numpy as np

class Script():
 
    def __init__(self):
        self.parser = None
        self.args = None
        self.out = None
        self.name='test'
        self.dim=None
        self.debug=False
        ######################### Init ########################
        
    def add_args(self,parser):
        parser.add_argument('--config', type=str, help='Load config from json file.')
        parser.add_argument('--seed', type=int, help='Set random\n' )
        parser.add_argument('--name', type=str, help='save model name\n')
        parser.add_argument('--out', type=str, help='output dir\n')
        parser.add_argument('--dim', type=int, default=None,  help='Latent Representation dimension\n')
        
    def create_parser(self):
        self.parser = argparse.ArgumentParser()
        self.add_args(self.parser)
    
    def parse_args(self):
        if self.args is None:
            self.args = self.parser.parse_args().__dict__
            self.get_configs()
        
    def get_configs(self):
        configs = []
        if 'config' in self.args and self.args['config'] is not None:
            configs = self.load_args_json(self.args['config'])
            for key, val in configs.items():
                if key not in self.args or self.args[key] is None:
                    self.args[key] = val
        else:
            raise "congif error"
    
    def load_args_json(self, filename):
        with open(filename, 'r') as f:
            args = json.load(f)
        return args
    
##################################################APPLY ARGS###############
        
    def apply_args(self):
        self.apply_init_args()
        self.apply_input_args()
        self.apply_output_args()
    
    def apply_init_args(self):
        if 'seed' in self.args and self.args['seed'] is not None:
            np.random.seed(self.args['seed'])  
        else:
            np.random.seed(112)  
        if 'name' in self.args and self.args['name'] is not None:
            self.name =self.args['name']

    def apply_input_args(self):
        if 'dim' in self.args and self.args['dim'] is not None:
            self.dim = self.args["dim"]
        else:
            raise "--dim latent dimension not specified"
  
    def apply_output_args(self):        
        if 'out' in self.args and self.args['out'] is not None:
            self.out = self.args['out'] 
            self.create_output_dir(self.out, cont=False)
        else:
            raise "--out output directory is not specified"
            
    def create_output_dir(self, dir, cont=False):
        logging.info('Output directory is {}'.format(dir))
        if cont:
            if os.path.exists(dir):
                logging.info('Found output directory.')
            else:
                raise Exception("Output directory doesn't exist, can't continue.")
        elif os.path.exists(dir):
            if len(os.listdir(dir)) != 0:
                print('Output directory not Empty, Replacing might occurs')
        else:
            logging.info('Creating output directory {}'.format(dir))
            os.makedirs(dir)
       

    def init_logging(self, outdir):
        self.setup_logging(os.path.join(outdir, type(self).__name__.lower() + '.log'))
        # handler = logging.StreamHandler(sys.stdout)
        # handler.setLevel(logging.DEBUG)
        # handler.setFormatter(formatter)
        # root.addHandler(handler)
    
    def setup_logging(self, logfile=None):
        root = logging.getLogger()
        root.setLevel(self.get_logging_level())
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    def get_logging_level(self):
        if self.debug: 
            return logging.DEBUG
        else: 
            return logging.INFO
            
    def prepare(self):
        self.create_parser()
        self.parse_args() 
        self.setup_logging()
        self.apply_args()
