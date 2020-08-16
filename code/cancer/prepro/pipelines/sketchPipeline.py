import logging
import os
import time
import copy
import math
import json
import random
import numpy as np

from cancer.script import Script

class CountSketchPipeline(Script):
    def __init__(self, logging=True):
        super().__init__()
        self.MODEL_TYPES = {
        'mode':{
            'exact': dense,
            'sketch': cnn,
        },
        'torchTL':{
            'dn121','dn161','dn169','dn201',    #densenet
            'rn18','rn34','rn50','rn101','rn152'    #resnet
        },
        'effi':{"b7"}
    }
        self.type = None
        self.method = None
        self.TL_mode = None
        self.model = None
        self.name = None
        self.n_epochs = 1
        self.opt = None
        self.lr = 0.01
        self.lrsch = None
        self.lrp = 0
        self.decay = 0.0009
        self.opt_name = "sgd"
        
        
    def add_args(self, parser):
        super().add_args(parser)
        parser.add_argument('--method', type=str, help='effi, torch or simpleNN\n')
        parser.add_argument('--type', type=str, help='model type\n')
        parser.add_argument('--TLfree', type=str, help='TL free: fc, half, full\n')
        parser.add_argument('--EFpt', type=str, help='pretrain model pt file\n')
        
        parser.add_argument('--ep', type=int, help='N epochs\n') 
        parser.add_argument('--opt', type=str, help='Optimizer\n')
        parser.add_argument('--lr', type=float, help='Learning rate\n')
        parser.add_argument('--lrsch', type=str, help='Learning scheduler\n')
        parser.add_argument('--decay', type=str, help='lr weight decay\n')
        parser.add_argument('--lrp', type=float, help='Optimizer momentum\n')

        
    def prepare(self):
        super().prepare()
        self.apply_model_args()

    def apply_model_args(self):
        if 'lr' in self.args and self.args['lr'] is not None:
            self.lr = self.args['lr']
        if 'lrp' in self.args and self.args['lrp'] is not None:
            self.lrp = self.args['lrp']
        if 'decay' in self.args and self.args['decay'] is not None:
            self.decay = self.args['decay']
        if 'ep' in self.args and self.args['ep'] is not None:
            self.n_epochs = self.args['ep']   
        if 'opt' in self.args and self.args['opt'] == 'adam':
            self.opt_name = 'adam'
        if 'type' in self.args and self.args['type'] is not None:
            self.type = self.args['type']
        if 'method' in self.args and self.args['method'] is not None:
            self.method = self.args['method']  
        if 'name' in self.args and self.args['name'] is not None:
            name = self.args['name'] 
        if self.method == 'simpleNN':
        ################# building simpleNN args #####################
            self.model = self.MODEL_TYPES[self.method][self.type]()
            print(f'Picking model: {self.method} {self.type}' )
            if self.opt_name == 'sgd':
                self.opt = optim.SGD(self.model.parameters(), lr=self.lr, momentum = self.lrp)
                print(f'Optimizer: SGD, lr {self.lr}')
            elif self.opt_name == 'adam':
                self.opt = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.decay)
                print(f'Optimizer: Adam, lr {self.lr}')
            if 'lrsch' in self.args and self.args['lrsch'] is not None:
                self.lrsch = self.args['lrsch']
            if self.on_cuda: self.model.cuda()
            self.name = f'{name}_{self.method}_{self.aug}_{self.opt_name}_lr{self.lr}_p{self.lrp}_ep{self.n_epochs}_bs{self.batch}' 
        elif self.method == 'torchTL':
        ################# building torchTL args #####################
            if 'lrsch' in self.args and self.args['lrsch'] is not None:
                self.lrsch = self.args['lrsch']
            if 'TLfree' in self.args and self.args['TLfree'] is not None:
                self.TLfree = self.args['TLfree']
            else:
                self.TLfree = "half"  
            
            self.name = f'{name}_{self.type}_{self.TLfree}_{self.opt_name}_lr{self.lr}' 

            get_torchTL(self)   
            # for name, param in self.model.named_parameters():
            #     print(name, param.requires_grad)          
            # for name, param in self.model.features.named_parameters():
            #     print(name, param.requires_grad)
            # for name, param in self.model.classifier.named_parameters():
            #     print(name, param.requires_grad)
        elif self.method == 'effi':
        ################# building effi args #####################
            if 'EFpt' in self.args and self.args['EFpt'] is not None:
                print(self.args['EFpt'])
                pt_file = self.args['EFpt']
            else:
                pt_file = None
            get_effinet(self, pt_file)
            self.name = f'{name}_{self.type}_{self.opt_name}_lr{self.lr}' 

        else:
            raise "--type not understood"
        print(f'Logdir Name: {self.name}')    

            

