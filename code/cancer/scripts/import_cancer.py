import sys
import os
import numpy as np
import pandas as pd
import argparse
# import umap
# import joblib
# import getpass
# print(os.getcwd())
sys.path.insert(0,'/home/swei20/cancerHH/AceCanZ/code/')

from cancer.prepro.pipelines.cellPipeline import CellPipeline


def main():
    p=CellPipeline()
    p.prepare()
    p.run()
# isTest=True
# isSmooth=False
# dataDir = r'./data/bki'  

main()
