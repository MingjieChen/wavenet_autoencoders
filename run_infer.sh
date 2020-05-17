#!/bin/bash
source activate torch_0.4
PYTHON=/share/mini1/sw/std/python/anaconda3-2019.07/v3.7/envs/torch_0.4/bin/python

$PYTHON inference_2017.py scp/2017/test_src_dst.json mfcc.norm 50 exp/0321vqvae_0/checkpoint_step000256000_ema.pth  exp/0321vqvae_0/infer/  english\
    --preset infer_hp.json 
                
