# coding: utf-8
"""
feature combination

usage: feat_combine.py [options]  <scp_dir> <infer_dir1> <infer_dir2> <dst_dir>

options:
    -h, --help                        Show help message.
"""
from docopt import docopt
import json
import sys
import os
from os.path import dirname, join, basename, splitext
import numpy as np
#import audio
#from hparams import hparams, hparams_debug_string
    
    
if __name__ == '__main__':
    args = docopt(__doc__)
    print("Command line args:\n", args)
    dst_dir = args["<dst_dir>"]
    os.makedirs(dst_dir, exist_ok=True)
    scp_dir = args['<scp_dir>']

    infer1_dir = args['<infer_dir1>']

    infer2_dir = args['<infer_dir2>']


    if not os.path.exists(infer1_dir):
        raise ValueError(f'{infer1_dir} does not exist')
    
    if not os.path.exists(infer2_dir):
        raise ValueError(f'{infer2_dir} does not exist')
    
    scp_f = open(scp_dir)
    file_list = json.load(scp_f)
    
    for _,base_dir in file_list:
        
        
        dirs = base_dir.split('/')
        assert len(dirs) == 6
        lan = dirs[-4]
        fnm = dirs[-2]

        feat1_dir = infer1_dir + f'2019/{lan}/test/{fnm}.txt'
        feat1 = np.loadtxt(feat1_dir)
        print(f"load feat1 shape {feat1.shape}",flush=True)

        feat2_dir = infer2_dir + f'2019/{lan}/test/{fnm}.txt'
        feat2 = np.loadtxt(feat2_dir)
        print(f"load feat2 shape {feat2.shape}",flush=True)

        new_feat = np.concatenate([feat1, feat2], axis = 1)
        
        new_feat_dir = f'{dst_dir}2019/{lan}/test/{fnm}.txt'

        os.makedirs(os.path.dirname(new_feat_dir),exist_ok=True)
        np.savetxt(new_feat_dir, new_feat, fmt='%.6f')
        print(f"new feat shape {new_feat.shape} {new_feat_dir}",flush=True)

