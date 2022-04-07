#!/bin/bash

source activate torch_0.4
PYTHON=/share/mini1/sw/std/python/anaconda3-2019.07/v3.7/envs/torch_0.4/bin/python


infer1=exp/0416vqvae_0/infer_350k/
infer2=exp/0323wvae_0/infer_500k/
dst=combine_feat/0416vqvae_0_0323wvae_0/
lan=english
scp=scp/2019/$lan/test_src_dst.json

$PYTHON feat_combine.py $scp $infer1 $infer2 $dst
