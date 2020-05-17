#!/bin/bash
source activate torch_0.4
PYTHON=/share/mini1/sw/std/python/anaconda3-2019.07/v3.7/envs/torch_0.4/bin/python

#$PYTHON inference_2019.py scp/2019/english/test_src_dst.json mfcc.norm exp/0329vqvae_1/checkpoint_step000160000_ema.pth  exp/0329vqvae_1/infer_160k/ \
#    --preset infer_hp.json 
                
exp=0511vqvae_0
step=checkpoint_step000350000_ema.pth
infer=exp/${exp}/infer_350k/
hp=exp/${exp}/hparams.json
lan=english

$PYTHON inference_2019.py scp/2019/$lan/test_src_dst.json mfcc.norm exp/$exp/$step  $infer  \
    --preset $hp 
