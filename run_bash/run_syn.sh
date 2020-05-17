#!/bin/bash
source activate torch_0.4
PYTHON=/share/mini1/sw/std/python/anaconda3-2019.07/v3.7/envs/torch_0.4/bin/python

#$PYTHON synthesis.py dump/2019/english/  exp/0322vqvae_0/checkpoint_step000288000_ema.pth  exp/0322vqvae_0/infer/  2020/2019/english/synthesis.txt 2019_speaker2ind.json english 160  25\
#    --preset syn_hp.json 
                
exp=0425vqvae_0
step=checkpoint_step000250000_ema.pth
#step=checkpoint_step000550000.pth
infer=exp/${exp}/infer_250k/
hp=exp/${exp}/hparams.json
fr=25
ind=0
lan=english

$PYTHON synthesis.py dump/2019/$lan/ exp/$exp/$step  $infer  2020/2019/$lan/synthesis.txt 2019_speaker2ind_${lan}.json $lan 160  $fr $ind \
    --preset $hp 
